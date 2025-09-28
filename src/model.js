import * as tf from '@tensorflow/tfjs';

export function linearRegressionModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [11], units: 1, useBias: true}));
    model.compile({
        optimizer: tf.train.adam(), 
        loss: 'meanSquaredError', 
        metrics: ['mae']});
    return model;
}