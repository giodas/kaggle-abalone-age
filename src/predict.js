import * as tf from '@tensorflow/tfjs-node';
import fs from 'node:fs';
import path from 'node:path';

// Paths
const DATA_DIR = path.resolve(process.cwd(), '../data');
const TEST_CSV_PATH = path.join(DATA_DIR, 'test.csv');
const TEST_CSV_URL = 'file://' + TEST_CSV_PATH;
console.log(TEST_CSV_URL);
const ARTIFACTS_DIR = path.resolve(process.cwd(), '../artifacts');
const MODEL_DIR = path.join(ARTIFACTS_DIR, 'abalone-linear-regression');
const MODEL_JSON_URL = 'file://' + path.join(MODEL_DIR, 'model.json');
const METADATA_PATH = path.join(ARTIFACTS_DIR, 'metadata.json');

function oneHotSex(sex, sexToIndex) {
  const idx = sexToIndex[sex];
  const v = [0, 0, 0];
  if (idx !== undefined) v[idx] = 1;
  return v;
}

async function loadArtifacts() {
  if (!fs.existsSync(METADATA_PATH)) throw new Error('metadata.json not found at ' + METADATA_PATH);
  if (!fs.existsSync(path.join(MODEL_DIR, 'model.json'))) throw new Error('model.json not found at ' + MODEL_DIR);
  const metadata = JSON.parse(fs.readFileSync(METADATA_PATH, 'utf-8'));
  const model = await tf.loadLayersModel(MODEL_JSON_URL);
  return { model, ...metadata };
}

function vectorize(xs, featureOrder, sexToIndex) {
  const sexOH = oneHotSex(xs.Sex, sexToIndex);
  const enriched = {
    ...xs,
    sex_M: sexOH[0],
    sex_F: sexOH[1],
    sex_I: sexOH[2]
  };
  delete enriched.Sex;
  return featureOrder.map(k => Number(enriched[k]));
}

// Test set has no target column; generate predictions only.
async function predictTestSet() {
  const { model, featureOrder, sexToIndex } = await loadArtifacts();
  if (!fs.existsSync(TEST_CSV_PATH)) throw new Error('test.csv not found in ' + DATA_DIR);

  const testDs = tf.data.csv(TEST_CSV_URL, { hasHeader: true });

  const outPath = path.resolve(process.cwd(), 'predictions.csv');
  const lines = ['id,Rings'];
  await testDs.forEachAsync(xs => {
    const row = vectorize(xs, featureOrder, sexToIndex);
    const x = tf.tensor2d([row]);
    const predT = model.predict(x);
    const pred = predT.dataSync()[0];
    lines.push(`${xs.id},${pred}`);
    x.dispose();
    predT.dispose();
  });

  fs.writeFileSync(outPath, lines.join('\n'));
  console.log(`Wrote predictions to ${outPath}`);
}

(async () => {
  try {
    await predictTestSet();
  } catch (e) {
    console.error(e);
    process.exit(1);
  }
})();
