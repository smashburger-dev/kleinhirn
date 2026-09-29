import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { HfTokenizer } from '../src/tokenizer/tokenizer.ts';
import { BucketOverflowError, prepareTasks } from '../src/tokenizer/schema.ts';

const MODELS = [
  'small-upstream', 'base-upstream', 'multi-upstream',
];

function loadJson(path: string) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

for (const model of MODELS) {
  const tokPath = `models/${model}/f32/tokenizer.json`;
  // models/ carries the (large, non-git) weights and tokenizer; a
  // checkout without it skips the golden checks instead of failing.
  const skip = existsSync(tokPath) ? false
    : `${tokPath} not present (models/ is not shipped)`;
  const tokenizer = skip ? null : new HfTokenizer(loadJson(tokPath));

  test(`${model}: tokenizer_ids golden`, { skip }, () => {
    const golden = loadJson(`tests/golden/${model}/tokenizer_ids.json`);
    let checked = 0;
    for (const item of golden.items) {
      const ids = tokenizer!.encodeIds(item.text);
      assert.deepEqual([...ids], item.ids, `text ${JSON.stringify(item.text)}`);
      checked += 1;
    }
    assert.equal(checked, golden.count);
  });

  test(`${model}: texts1000 schema input`, { skip }, () => {
    const golden = loadJson(`tests/golden/${model}/texts1000_l128k16.json`);
    const tasks: [string, string[]][] = [[golden.task, golden.labels]];
    for (const item of golden.items) {
      const out = prepareTasks(tokenizer!, item.title, tasks, 128, 16);
      assert.deepEqual(
        [...out.inputIds.slice(0, out.seqLen)], item.input_ids, item.title);
      assert.deepEqual(
        [...out.markerIndices.slice(0, item.marker_indices.length)],
        item.marker_indices, item.title);
      const n = item.marker_indices.length;
      assert.deepEqual(
        [...out.markerMask.slice(0, n)], item.marker_mask.slice(0, n));
      assert.deepEqual(
        [...out.markerGroups.slice(0, n)], item.marker_groups.slice(0, n));
    }
  });

  test(`${model}: two-task schema input`, { skip }, () => {
    const golden = loadJson(`tests/golden/${model}/two_tasks_l128k16.json`);
    const tasks: [string, string[]][] = Object.entries(golden.tasks);
    for (const item of golden.items) {
      const out = prepareTasks(tokenizer!, item.title, tasks, 128, 16);
      assert.deepEqual(
        [...out.inputIds.slice(0, out.seqLen)], item.input_ids, item.title);
      const n = item.marker_indices.length;
      assert.deepEqual([...out.markerIndices.slice(0, n)], item.marker_indices);
      assert.deepEqual([...out.markerGroups.slice(0, n)], item.marker_groups);
    }
  });

  test(`${model}: bucket overflow throws`, { skip }, () => {
    const tasks: [string, string[]][] = [['event', ['a', 'b', 'c']]];
    assert.throws(
      () => prepareTasks(tokenizer!, 'x '.repeat(500), tasks, 128, 16),
      BucketOverflowError);
    assert.throws(
      () => prepareTasks(tokenizer!, 'hi', [['a', Array(20).fill('l')]], 128, 16),
      BucketOverflowError);
  });
}
