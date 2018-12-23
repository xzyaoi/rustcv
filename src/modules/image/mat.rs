extern crate image;

pub type ImageData = Matrix<f64>;
pub type Label = f64;

#[derive(Debug)]
pub struct Mat {
    pub image_data ImageData,
    pub label: Label,
}