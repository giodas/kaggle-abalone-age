import * as tf from '@tensorflow/tfjs-node';
import * as model from './model.js';
import fs from 'node:fs';
import path from 'node:path';

const DATA_BASE_URL= 'file://../data/';
const TRAIN_DATA = 'train.csv';

// This constant defines a lookup table mapping each categorical Sex code to a stable 
// numeric index: 'M' → 0, 'F' → 1, 'I' (infant) → 2. Such a mapping is a common 
// preprocessing step so downstream code can convert human‑readable category labels 
// into numeric positions for arrays or tensors (e.g., for one‑hot encoding or embedding lookups).
const SEX_TO_INDEX = { M: 0, F: 1, I: 2 };

// This helper performs one‑hot encoding for the categorical Sex field. 
// It first looks up the numeric index associated with the incoming sex code (e.g. 'M', 'F', 'I') in the constant mapping SEX_TO_INDEX. 
// That index (0, 1, or 2) designates which position in a fixed-length vector should be 1 while the others are 0.
function oneHotSex(sex) {
  const idx = SEX_TO_INDEX[sex];
  // Return plain JS array; tf.data will later convert to tensor
  return [idx === 0 ? 1 : 0, idx === 1 ? 1 : 0, idx === 2 ? 1 : 0];
}

async function trainLinear(trainDataSet) {
  // Determine feature order from first sample (without materializing entire dataset)
  let featureOrder = null;
  let firstRowVector = null;
  await trainDataSet.take(1).forEachAsync(s => {
    featureOrder = Object.keys(s.xs);
    firstRowVector = featureOrder.map(k => Number(s.xs[k]));
  });
  if (!featureOrder) throw new Error('No data.');

  // Materialize entire dataset into memory (dataset is small) for model.fit with validationSplit.
  const featureRows = [];
  const labelRows = [];
  await trainDataSet.forEachAsync(({ xs, ys }) => {
    const row = featureOrder.map(k => Number(xs[k]));
    featureRows.push(row);
    labelRows.push(Number(ys.Rings));
  });

  if (featureRows.length === 0) throw new Error('No rows collected from dataset.');

  const xTensor = tf.tensor2d(featureRows);
  const yTensor = tf.tensor2d(labelRows, [labelRows.length, 1]);

  const linearRegressionModel = model.linearRegressionModel(featureOrder.length);
  linearRegressionModel.summary();

  linearRegressionModel.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  const logDir = path.resolve(process.cwd(), '../logs/fit');
  const tbCallback = tf.node.tensorBoard(logDir);

  await linearRegressionModel.fit(xTensor, yTensor, {
    epochs: 10,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.1,
    verbose: 1,
    callbacks: [tbCallback]
  });

  // Quick sanity prediction on first row (built earlier)
  const firstX = tf.tensor2d([firstRowVector]);
  const pred = linearRegressionModel.predict(firstX).dataSync()[0];
  console.log('First row prediction (no ground truth here after streaming):', pred);
  firstX.dispose();

  // --- Save artifacts (Node only) ---
  const artifactsDir = path.resolve(process.cwd(), '../artifacts');
  if (!fs.existsSync(artifactsDir)) fs.mkdirSync(artifactsDir, { recursive: true });
  const modelDir = path.join(artifactsDir, 'abalone-linear-regression');
  await linearRegressionModel.save(`file://${modelDir}`);
  const metadata = { featureOrder, sexToIndex: SEX_TO_INDEX, createdAt: new Date().toISOString() };
  fs.writeFileSync(path.join(artifactsDir, 'metadata.json'), JSON.stringify(metadata, null, 2));
  console.log('Saved model to', modelDir);
  console.log('Saved metadata to artifacts/metadata.json');
  // ----------------------------------

  xTensor.dispose();
  yTensor.dispose();

  return { linearRegressionModel, featureOrder };
}

async function run() {
  const rawTrainDataSet = tf.data.csv(DATA_BASE_URL + TRAIN_DATA, {
    hasHeader: true,
    columnConfigs: { Rings: { isLabel: true } }
  });

  // Expand with one-hot (object form) for initial peek & feature order derivation
  const trainDataSet = rawTrainDataSet.map(({ xs, ys }) => {
    const sexOH = oneHotSex(xs.Sex);
    delete xs.Sex;
    return {
      xs: {
        ...xs,
        sex_M: sexOH[0],
        sex_F: sexOH[1],
        sex_I: sexOH[2]
      },
      ys
    };
  });

  await trainDataSet.take(1).forEachAsync(d => console.log('Sample after encoding (preview):', d));

  await trainLinear(trainDataSet);
}

run().catch(console.error);