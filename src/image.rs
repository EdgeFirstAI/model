// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use core::fmt;
use dma_buf::DmaBuf;
use dma_heap::{Heap, HeapKind};
use four_char_code::{FourCharCode, four_char_code};
use g2d_sys::{
    G2D, G2DFormat, G2DPhysical, G2DSurface, g2d_format_G2D_RGB888, g2d_format_G2D_RGBX8888,
    g2d_rotation_G2D_ROTATION_0, g2d_rotation_G2D_ROTATION_90, g2d_rotation_G2D_ROTATION_180,
    g2d_rotation_G2D_ROTATION_270,
};
use log::{debug, warn};
use nix::libc::{MAP_SHARED, PROT_READ, PROT_WRITE, dup, mmap, munmap};
use std::{
    error::Error,
    ffi::c_void,
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd},
        unix::io::OwnedFd,
    },
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub const RGB3: FourCharCode = four_char_code!("RGB3");
pub const RGBX: FourCharCode = four_char_code!("RGBX");
pub const RGBA: FourCharCode = four_char_code!("RGBA");
pub const YUYV: FourCharCode = four_char_code!("YUYV");
pub const NV12: FourCharCode = four_char_code!("NV12");

pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
pub enum Rotation {
    Rotation0 = g2d_rotation_G2D_ROTATION_0 as isize,
    Rotation90 = g2d_rotation_G2D_ROTATION_90 as isize,
    Rotation180 = g2d_rotation_G2D_ROTATION_180 as isize,
    Rotation270 = g2d_rotation_G2D_ROTATION_270 as isize,
}

/// Convert a FourCharCode to a g2d_format, supporting formats not in the
/// upstream G2DFormat::try_from (RGB3 and RGBX).
fn g2d_format_from_fourcc(fourcc: FourCharCode) -> Result<g2d_sys::g2d_format, Box<dyn Error>> {
    match fourcc {
        RGB3 => Ok(g2d_format_G2D_RGB888),
        RGBX => Ok(g2d_format_G2D_RGBX8888),
        _ => Ok(G2DFormat::try_from(fourcc)?.format()),
    }
}

/// Create a G2DSurface from an Image.
fn surface_from_image(img: &Image) -> Result<G2DSurface, Box<dyn Error>> {
    let fd = img.fd.try_clone()?;
    let phys = G2DPhysical::new(fd.as_raw_fd())?;
    let format = g2d_format_from_fourcc(img.format)?;

    Ok(G2DSurface {
        planes: [phys.address(), 0, 0],
        format,
        left: 0,
        top: 0,
        right: img.width as i32,
        bottom: img.height as i32,
        stride: img.width as i32,
        width: img.width as i32,
        height: img.height as i32,
        ..G2DSurface::default()
    })
}

pub struct ImageManager {
    g2d: G2D,
}

impl ImageManager {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let g2d = G2D::new("libg2d.so.2")?;
        debug!("G2D version: {}", g2d.version());
        Ok(Self { g2d })
    }

    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }

    pub fn convert(
        &self,
        from: &Image,
        to: &Image,
        crop: Option<Rect>,
        rot: Rotation,
    ) -> Result<(), Box<dyn Error>> {
        let mut src = surface_from_image(from)?;

        if let Some(r) = crop {
            src.left = r.x;
            src.top = r.y;
            src.right = r.x + r.width;
            src.bottom = r.y + r.height;
        }

        let mut dst = surface_from_image(to)?;
        dst.rot = rot as u32;

        self.g2d.blit(&src, &dst)?;
        self.g2d.finish()?;
        // FIXME: A cache invalidation is required here, currently missing!

        Ok(())
    }
}

impl Drop for ImageManager {
    fn drop(&mut self) {
        debug!("G2D closed");
    }
}

#[derive(Debug)]
pub struct Image {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
    pub format: FourCharCode,
}

const fn format_row_stride(format: FourCharCode, width: u32) -> usize {
    match format {
        RGB3 => 3 * width as usize,
        RGBX => 4 * width as usize,
        RGBA => 4 * width as usize,
        YUYV => 2 * width as usize,
        NV12 => width as usize / 2 + width as usize,
        _ => todo!(),
    }
}

const fn image_size(width: u32, height: u32, format: FourCharCode) -> usize {
    format_row_stride(format, width) * height as usize
}

impl Image {
    pub fn new(width: u32, height: u32, format: FourCharCode) -> Result<Self, Box<dyn Error>> {
        let heap = Heap::new(HeapKind::Cma)?;
        let fd = heap.allocate(image_size(width, height, format))?;
        Ok(Self {
            fd,
            width,
            height,
            format,
        })
    }

    pub fn new_preallocated(fd: OwnedFd, width: u32, height: u32, format: FourCharCode) -> Self {
        Self {
            fd,
            width,
            height,
            format,
        }
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }

    pub fn dmabuf(&self) -> DmaBuf {
        unsafe { DmaBuf::from_raw_fd(dup(self.fd.as_raw_fd())) }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> FourCharCode {
        self.format
    }

    pub fn size(&self) -> usize {
        format_row_stride(self.format, self.width) * self.height as usize
    }

    pub fn mmap(&mut self) -> MappedImage {
        let image_size = image_size(self.width, self.height, self.format);
        unsafe {
            let mmap = mmap(
                null_mut(),
                image_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.raw_fd(),
                0,
            ) as *mut u8;
            MappedImage {
                mmap,
                len: image_size,
            }
        }
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}x{} {} fd:{:?}",
            self.width,
            self.height,
            self.format.display(),
            self.fd
        )
    }
}

pub struct MappedImage {
    mmap: *mut u8,
    len: usize,
}

impl MappedImage {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { from_raw_parts(self.mmap, self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { from_raw_parts_mut(self.mmap, self.len) }
    }
}
impl Drop for MappedImage {
    fn drop(&mut self) {
        if unsafe { munmap(self.mmap.cast::<c_void>(), self.len) } > 0 {
            warn!("unmap failed!");
        }
    }
}
