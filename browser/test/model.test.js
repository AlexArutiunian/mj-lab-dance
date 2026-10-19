import test from 'node:test';
import assert from 'node:assert/strict';
import {alphaRef,healthState,rot6,torsoQuat} from '../src/model.js';
test('identity orientation',()=>{assert.deepEqual(rot6([1,0,0,0],[1,0,0,0],[1,0,0,0]),[1,0,0,1,0,0]);assert.deepEqual(torsoQuat([1,0,0,0],Array(29).fill(0)),[1,0,0,0]);});
test('reference damage and floor',()=>{const a=healthState(1e6,10,alphaRef/10);assert.ok(Math.abs(a.u-0.3)<1e-12);assert.equal(healthState(1e9,10,alphaRef/10).u,0.1);assert.equal(healthState(0,10,alphaRef/10).u,1);});
