// Memory-light CPU execution of the original exported ONNX graph.
// Intended for iOS Safari where two WebAssembly runtimes exceed WebKit limits.
// Model parameters and graph come directly from the original ONNX model at build time.
const product=d=>d.reduce((a,b)=>a*b,1);
const tensor=(data,shape)=>({data,shape});
function broadcast(a,b,fn){
  const len=Math.max(a.data.length,b.data.length);
  const out=new Float32Array(len);
  if(a.data.length!==len && a.data.length!==1 && len%a.data.length!==0)
    throw Error('Unsupported ONNX broadcast A: '+a.shape);
  if(b.data.length!==len && b.data.length!==1 && len%b.data.length!==0)
    throw Error('Unsupported ONNX broadcast B: '+b.shape);
  for(let i=0;i<len;i++){
    const ai=a.data.length===1?0:i%a.data.length;
    const bi=b.data.length===1?0:i%b.data.length;
    out[i]=fn(a.data[ai],b.data[bi]);
  }
  return tensor(out,a.data.length>=b.data.length?a.shape:b.shape);
}
function matmul(a,b,gemm=false,attrs={}){
  if(a.shape.length!==2||b.shape.length!==2)
    throw Error('Only rank-2 ONNX MatMul/Gemm is supported');
  const ta=Number(attrs.transA||0),tb=Number(attrs.transB||0);
  const m=ta?a.shape[1]:a.shape[0];
  const k=ta?a.shape[0]:a.shape[1];
  const kb=tb?b.shape[1]:b.shape[0];
  const n=tb?b.shape[0]:b.shape[1];
  if(k!==kb)throw Error('ONNX matrix mismatch: '+a.shape+' / '+b.shape);
  const out=new Float32Array(m*n);
  const alpha=gemm?(attrs.alpha??1):1;
  for(let i=0;i<m;i++){
    for(let j=0;j<n;j++){
      let sum=0;
      for(let t=0;t<k;t++){
        const aa=ta?a.data[t*m+i]:a.data[i*k+t];
        const bb=tb?b.data[j*k+t]:b.data[t*n+j];
        sum+=aa*bb;
      }
      out[i*n+j]=alpha*sum;
    }
  }
  return tensor(out,[m,n]);
}
function elem(a,fn){
  const out=new Float32Array(a.data.length);
  for(let i=0;i<out.length;i++)out[i]=fn(a.data[i]);
  return tensor(out,a.shape);
}
function transpose(t,perm){
  const rank=t.shape.length;
  const p=perm??Array.from({length:rank},(_,i)=>rank-1-i);
  const shape=p.map(i=>t.shape[i]),out=new Float32Array(t.data.length);
  const strides=d=>d.map((_,i)=>product(d.slice(i+1)));
  const from=strides(t.shape),to=strides(shape);
  for(let i=0;i<out.length;i++){
    let rem=i,index=0;
    for(let d=0;d<rank;d++){
      const coord=Math.floor(rem/to[d]);rem%=to[d];index+=coord*from[p[d]];
    }
    out[i]=t.data[index];
  }
  return tensor(out,shape);
}
function concat(parts,axis){
  const outShape=parts[0].shape.slice(),rank=outShape.length;
  if(axis<0)axis+=rank;
  outShape[axis]=parts.reduce((a,t)=>a+t.shape[axis],0);
  const out=new Float32Array(product(outShape));
  const block=product(outShape.slice(axis+1)),prefix=product(outShape.slice(0,axis));
  for(let p=0;p<prefix;p++){
    let dest=p*outShape[axis]*block;
    for(const t of parts){
      const size=t.shape[axis]*block;
      out.set(t.data.subarray(p*size,(p+1)*size),dest);dest+=size;
    }
  }
  return tensor(out,outShape);
}
function evaluate(node,vars){
  const op=node.op, attrs=node.attrs||{};
  const t=node.inputs.map(name=>name?vars.get(name):null);
  const a=t[0],b=t[1];
  const binary={Add:(x,y)=>x+y,Sub:(x,y)=>x-y,Mul:(x,y)=>x*y,
    Div:(x,y)=>x/y,Pow:(x,y)=>Math.pow(x,y)};
  if(binary[op])return broadcast(a,b,binary[op]);
  if(op==='Gemm'){
    const result=matmul(a,b,true,attrs);
    if(t[2])return broadcast(result,elem(t[2],v=>v*(attrs.beta??1)),binary.Add);
    return result;
  }
  if(op==='MatMul')return matmul(a,b);
  if(op==='Relu')return elem(a,v=>Math.max(0,v));
  if(op==='Tanh')return elem(a,Math.tanh);
  if(op==='Sigmoid')return elem(a,v=>1/(1+Math.exp(-v)));
  if(op==='Elu')return elem(a,v=>v>=0?v:(attrs.alpha??1)*Math.expm1(v));
  if(op==='LeakyRelu')return elem(a,v=>v>=0?v:(attrs.alpha??.01)*v);
  if(op==='Neg')return elem(a,v=>-v);
  if(op==='Exp')return elem(a,Math.exp);
  if(op==='Sqrt')return elem(a,Math.sqrt);
  if(op==='Identity'||op==='Cast')return a;
  if(op==='Flatten'){
    let axis=attrs.axis??1;if(axis<0)axis+=a.shape.length;
    return tensor(a.data,[product(a.shape.slice(0,axis)),product(a.shape.slice(axis))]);
  }
  if(op==='Reshape'){
    const shape=Array.from(b.data,Number);
    for(let i=0;i<shape.length;i++){
      if(shape[i]===0)shape[i]=a.shape[i];
      if(shape[i]===-1)shape[i]=a.data.length/product(shape.filter(x=>x>0));
    }
    if(product(shape)!==a.data.length)throw Error('Invalid ONNX Reshape');
    return tensor(a.data,shape);
  }
  if(op==='Squeeze'){
    const axes=b?Array.from(b.data,Number):(attrs.axes||[]);
    const shape=a.shape.filter((s,i)=>!axes.includes(i));
    return tensor(a.data,shape);
  }
  if(op==='Unsqueeze'){
    const axes=b?Array.from(b.data,Number):(attrs.axes||[]);
    const shape=a.shape.slice();
    axes.slice().sort((x,y)=>x-y).forEach(i=>shape.splice(i,0,1));
    return tensor(a.data,shape);
  }
  if(op==='Transpose')return transpose(a,attrs.perm);
  if(op==='Concat')return concat(t,attrs.axis??0);
  if(op==='Clip'){
    const min=b?b.data[0]:(attrs.min??-Infinity);
    const max=t[2]?t[2].data[0]:(attrs.max??Infinity);
    return elem(a,v=>Math.max(min,Math.min(max,v)));
  }
  throw Error('Unsupported JS ONNX operation: '+op);
}
export async function loadJsPolicy(base){
  const [metaResponse,weightResponse]=await Promise.all([
    fetch(new URL('policy.js.json',base)),
    fetch(new URL('policy.js.bin',base))
  ]);
  if(!metaResponse.ok||!weightResponse.ok)
    throw Error('JS ONNX assets missing: '+metaResponse.status+'/'+weightResponse.status);
  const [meta,buf]=await Promise.all([metaResponse.json(),weightResponse.arrayBuffer()]);
  if(meta.format!=='onnx-fp32-graph-v1'||buf.byteLength!==meta.float_count*4)
    throw Error('Invalid JS ONNX policy format or weights');
  const all=new Float32Array(buf);
  const weights=new Map();
  for(const [name,info] of Object.entries(meta.constants)){
    if(info.offset<0||info.offset+info.length>all.length)throw Error('ONNX weight range');
    weights.set(name,tensor(all.subarray(info.offset,info.offset+info.length),info.shape));
  }
  const inputNames=meta.inputs,outputNames=meta.outputs;
  return {inputNames,outputNames,backend:'js-fp32',
    async run(feeds){
      const input=feeds[inputNames[0]];
      if(!input?.data)throw Error('Missing ONNX observation');
      const vars=new Map(weights);
      vars.set(inputNames[0],tensor(input.data,[1,input.data.length]));
      for(const node of meta.nodes){
        if(node.outputs.length!==1)throw Error('Unsupported ONNX multiple outputs: '+node.op);
        vars.set(node.outputs[0],evaluate(node,vars));
      }
      const out=vars.get(outputNames[0]);
      if(!out)throw Error('ONNX output missing');
      return {[outputNames[0]]:{data:out.data}};
    }
  };
}
