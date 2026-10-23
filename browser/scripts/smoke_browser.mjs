import {chromium} from 'playwright';

const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1200,height:750}});
const errors=[];
page.on('pageerror',e=>errors.push('page '+e.message));
page.on('console',m=>{if(m.type()==='error')errors.push('console '+m.text());});
page.on('requestfailed',q=>errors.push('network '+q.url()+' '+q.failure()?.errorText));
page.on('response',r=>{if(r.status()>=400)errors.push('HTTP '+r.status()+' '+r.url());});

const waitRunning=async()=>{
  await page.waitForFunction(()=>document.querySelector('#verdict').textContent==='Выполняется'
    && parseFloat(document.querySelector('#time').textContent)>0.1,
    null,{timeout:120000});
};
const getButtons=async()=>page.locator('#preset-buttons button').allTextContents();
const expected={
  short:['300 тыс.','500 тыс.','850 тыс.'],
  full:['10 тыс.','50 тыс.','125 тыс.'],
  walk:['500 тыс.','1 млн','1,5 млн']
};
const selectMotion=async motion=>{
  await page.locator('#motion').selectOption(motion);
  await page.waitForFunction(()=>document.querySelector('#repeats').value==='0',null,{timeout:5000});
  const buttons=await getButtons();
  if(JSON.stringify(buttons)!==JSON.stringify(expected[motion]))
    throw new Error('Incorrect three shortcuts for '+motion+': '+buttons);
  await waitRunning();
  console.log('AUTO HEALTHY',motion,await page.locator('#time').textContent(),buttons.join(' / '));
};

try{
  await page.goto('http://127.0.0.1:4173/',{waitUntil:'domcontentloaded',timeout:30000});
  await page.waitForFunction(()=>document.querySelectorAll('#preset-buttons button').length===3,null,{timeout:10000});
  const startR=await page.locator('#repeats').inputValue();
  if(startR!=='0')throw new Error('Expected zero initial repetitions, received '+startR);
  await waitRunning();
  console.log('INITIAL AUTOPLAY R=0 OK');
  if(JSON.stringify(await getButtons())!==JSON.stringify(expected.short))
    throw new Error('Default three wear shortcuts are incorrect');

  const mode=page.locator('#camera-mode');
  for(const name of ['torso','free','follow']){
    await mode.click();
    if(await mode.getAttribute('data-mode')!==name)
      throw new Error('Camera toggle failed for '+name);
  }
  console.log('THREE CAMERA MODES OK');

  for(const motion of ['full','walk','short'])await selectMotion(motion);

  // Presets launch a real fresh rollout rather than only changing text fields.
  await page.locator('#preset-buttons button[data-repeats="500000"]').click();
  await page.waitForFunction(()=>Number(document.querySelector('#health').textContent)<1 &&
    document.querySelector('#repeats').value==='500000' &&
    parseFloat(document.querySelector('#time').textContent)>.1,
    null,{timeout:180000});
  console.log('SHORT PRESET: simulated R=500000, health',await page.locator('#health').textContent());

  // Walking has its own 98-float policy observation and must compute its own
  // 60-second healthy work profile before nonzero-R torque degradation.
  await selectMotion('walk');
  await page.locator('#preset-buttons button[data-repeats="500000"]').click();
  await page.waitForFunction(()=>Number(document.querySelector('#health').textContent)<1 &&
    document.querySelector('#repeats').value==='500000' &&
    parseFloat(document.querySelector('#time').textContent)>.1,
    null,{timeout:300000});
  console.log('WALK PRESET: native-style 98D ONNX, R=500000, health',await page.locator('#health').textContent());

  // The previous passive fall sequence and camera tracking must still work.
  await selectMotion('short');
  await page.locator('#repeats').fill('100000000');
  await page.locator('#play').click();  // pause autoplay
  await page.waitForFunction(()=>document.querySelector('#verdict').textContent==='Пауза',null,{timeout:10000});
  await page.locator('#play').click();  // resumes existing rollout
  // Manually changed R should start a new rollout using the Run button.
  await page.locator('#reset').click();
  await page.locator('#repeats').fill('100000000');
  await page.locator('#play').click();
  await page.waitForFunction(()=>document.querySelector('#verdict').textContent.includes('приводы отключены'),null,{timeout:180000});
  const fallTime=parseFloat(await page.locator('#time').textContent());
  await page.waitForFunction(()=>
    document.querySelector('#verdict').textContent.includes('управление отключено') ||
    document.querySelector('#verdict').textContent.includes('расчёт завершён'),
    null,{timeout:120000});
  const finalTime=parseFloat(await page.locator('#time').textContent());
  if(finalTime-fallTime<1)throw Error('Unpowered MuJoCo dynamics did not continue');
  console.log('UNPOWERED FALL AND REST OK',finalTime-fallTime,'s');
  await page.locator('#reset').click();
  await page.waitForFunction(()=>document.querySelector('#time').textContent==='0.00 с'
    && document.querySelector('#repeats').value==='0',null,{timeout:10000});
  console.log('RESET R=0 OK');
  await page.screenshot({path:'smoke.png',fullPage:true});
}catch(err){
  console.error('BROWSER SMOKE ERROR:',err.message);
  console.error('Status:',await page.locator('#stage-status').textContent());
  console.error('Message:',await page.locator('#message').textContent());
  console.error('Other errors:',errors.slice(0,30));
  await page.screenshot({path:'smoke-failed.png',fullPage:true});
  process.exitCode=1;
}finally{await browser.close();}
