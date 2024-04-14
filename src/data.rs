use common_models::time_series::{GruNetwork, GruNetworkConfig};
use ml_boilerplate::burn::{
    self,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Data, Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use nalgebra::Vector3;
use rand::prelude::SliceRandom;
use rand::{distributions::Distribution, thread_rng, Rng};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

const SEQ_LENGTH: usize = 20;
const STD_DEV_COUNT: usize = 2;
const MAX_STD_DEV: f32 = 0.35;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct TrainingItem {
    #[serde(with = "BigArray")]
    input: [(Vector3<f32>, f32); SEQ_LENGTH],
    target: Vector3<f32>,
}

pub fn gen() -> TrainingItem {
    let mut rng = thread_rng();
    let mut item = TrainingItem {
        input: [(Vector3::default(), 0.0); SEQ_LENGTH],
        target: Vector3::default(),
    };

    let mut origin = Vector3::default();
    let mut velocity = Vector3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );

    let mut i = 0usize;
    let mut std_devs = vec![Normal::new(0.0, 0.0).unwrap()];
    for _ in 0..STD_DEV_COUNT {
        std_devs.push(Normal::new(0.0, rng.gen_range(0.0..MAX_STD_DEV)).unwrap());
    }
    while i < SEQ_LENGTH {
        let next_velocity = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        let seq_length = rng.gen_range(0..SEQ_LENGTH);
        let accel = (next_velocity - velocity).unscale(seq_length as f32);
        for _ in 0..seq_length {
            let normal = std_devs.choose(&mut rng).unwrap();

            origin += velocity + accel / 2.0;
            velocity += accel;

            item.input[i] = (
                origin
                    + Vector3::new(
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                    ),
                normal.std_dev(),
            );
            item.target = velocity;

            i += 1;
            if i >= SEQ_LENGTH {
                break;
            }
        }
    }

    // item.target = origin;
    item
}

#[derive(Clone)]
pub struct TrainingBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TrainingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,
    pub target: Tensor<B, 2>,
}

impl<B: Backend> Batcher<TrainingItem, TrainingBatch<B>> for TrainingBatcher<B> {
    fn batch(&self, items: Vec<TrainingItem>) -> TrainingBatch<B> {
        let inputs = items
            .iter()
            .map(|item| {
                Data::<f32, 2>::from(
                    item.input
                        .map(|(origin, dev)| [origin.x, origin.y, origin.z, dev]),
                )
            })
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, SEQ_LENGTH, 4]))
            .collect();

        let target = items
            .iter()
            .map(|item| {
                Tensor::<B, 2>::from_data(
                    Data::<f32, 2>::from([[item.target.x, item.target.y, item.target.z]]).convert(),
                    &self.device,
                )
            })
            .map(|tensor| tensor.reshape([1, 3]))
            .collect();

        let inputs = Tensor::cat(inputs, 0).to_device(&self.device);
        let target = Tensor::cat(target, 0).to_device(&self.device);

        TrainingBatch { inputs, target }
    }
}

#[derive(Module, Debug)]
pub struct TrainingModel<B: Backend> {
    model: GruNetwork<B>,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingModelConfig(GruNetworkConfig);

impl Config for TrainingModelConfig {}

impl TrainingModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrainingModel<B> {
        TrainingModel {
            model: self.0.init(device),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, RegressionOutput<B>> for TrainingModel<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.model.forward(batch.inputs);
        let loss = MseLoss::new().forward(output.clone(), batch.target.clone(), Reduction::Mean);
        let item = RegressionOutput::new(loss, output, batch.target);

        TrainOutput::new(&self.model, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TrainingBatch<B>, RegressionOutput<B>> for TrainingModel<B> {
    fn step(&self, batch: TrainingBatch<B>) -> RegressionOutput<B> {
        let output = self.model.forward(batch.inputs);
        let loss = MseLoss::new().forward(output.clone(), batch.target.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, batch.target)
    }
}
