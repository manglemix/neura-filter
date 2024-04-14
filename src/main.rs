mod data;
use data::{TrainingItem, TrainingBatcher, gen};
use ml_boilerplate::ml_app;

use crate::data::TrainingModelConfig;


fn main() {
    ml_app!(TrainingBatcher, TrainingItem, gen, TrainingModelConfig);
}
