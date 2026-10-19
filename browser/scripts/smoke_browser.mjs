import { chromium } from 'playwright';
const browser=await chromium.launch({
  headless:true,args:['--enable-unsafe-swiftshader','--disable-dev-shm-usage']
});
const page=await browser.newPage({viewport:{width:1200,height:750}});
const errors=[];
page.on('pageerror',e=>errors.push('page error '+e.message));
page.on('requestfailed',req=>errors.push('network '+req.url()+' '+req.failure()?.errorText));
page.on('response',res=>{if(res.status()>=400)errors.push('HTTP '+res.status()+' '+res.url())});
page.on('console',m=>{if(m.type()==='error')errors.push('console '+m.text());});
try{
  await page.goto('http://127.0.0.1:4173/',{waitUntil:'domcontentloaded',timeout:30000});
  await page.waitForFunction(()=>!document.querySelector('#play').disabled,{timeout:120000});
  console.log('G1 model loaded');
  await page.locator('#play').click();
  await page.waitForFunction(()=>parseFloat(document.querySelector('#time').textContent)>0.1,{timeout:120000});
  console.log('ONNX policy and physics advanced:',await page.locator('#time').textContent());
  console.log('Health:',await page.locator('#health').textContent());
  await page.screenshot({path:'smoke.png',fullPage:true});
} catch(error){
  console.error('BROWSER SMOKE ERROR:',error.message);
  console.error('Status:',await page.locator('#stage-status').textContent());
  console.error('Message:',await page.locator('#message').textContent());
  console.error('Other errors:',errors.slice(0,30));
  await page.screenshot({path:'smoke-failed.png',fullPage:true});
  process.exitCode=1;
} finally{await browser.close();}
