extern crate rand;

use self::rand::distributions::{Range, IndependentSample};

use std::cell::RefCell;
use std::rc::Rc;

use layer::NetworkLayer;

pub struct FullyConnectedLayer {
    prev: Rc<RefCell<Box<NetworkLayer>>>,
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    raw_outputs: Vec<f64>,
    outputs: Vec<f64>,
    errors: Vec<f64>,
    n: usize
}

impl FullyConnectedLayer {
    pub fn new(n: usize, prev: &Rc<RefCell<Box<NetworkLayer>>>) -> FullyConnectedLayer {
        let m = prev.borrow().num_outputs();
        let mut layer = FullyConnectedLayer {
            prev: prev.clone(),
            weights: vec![vec![0.0;n];m],
            bias: vec![0.0;n],
            raw_outputs: vec![0.0;n],
            errors: vec![0.0;n],
            n:n
        };
        let range = Range::new(-1.0 / (m as f64).sqrt(), 1.0 / (m as f64).sqrt());
        
    }
}