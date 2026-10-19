import {chromium} from 'playwright';
const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
const page=await browser.newPage({viewport:{width:1200,height:750}});
const errors=[];
page.on('pageerror',e=>errors.push('page '+e.message));
page.on('console',m=>{if(m.type()==='error')errors.push('console '+m.text());});
page.on('requestfailed',q=>errors.push('network '+q.url()+' '+q.failure()?.errorText));
page.on('response',r=>{if(r.status()>=400)errors.push('HTTP '+r.status()+' '+r.url());});
try{
  await page.goto('http://127.0.0.1:4173/',{waitUntil:'domcontentloaded',timeout:30000});
  await page.waitForFunction(()=>!document.querySelector('#play').disabled,{timeout:120000});
  console.log('G1 model ready');
  for(const mode of ['short','full','clip10']){
    await page.locator('#motion').selectOption(mode);
    await page.locator('#repeats').fill('0');
    await page.locator('#play').click();
    await page.waitForFunction(()=>parseFloat(document.querySelector('#time').textContent)>0.1,
      {timeout:120000});
    console.log('HEALTHY',mode,await page.locator('#time').textContent());
    await page.locator('#reset').click();
    await page.waitForTimeout(300);
  }
  await page.locator('#motion').selectOption('short');
  await page.locator('#repeats').fill('500000');
  await page.locator('#play').click();
  await page.waitForFunction(()=>parseFloat(document.querySelector('#time').textContent)>0.1 &&
    parseFloat(document.querySelector('#health').textContent)<1,{timeout:180000});
  console.log('AGED short: time',await page.locator('#time').textContent(),
    'health',await page.locator('#health').textContent(),
    'torque',await page.locator('#torque').textContent());
  await page.screenshot({path:'smoke.png',fullPage:true});
} catch(error){
  console.error('BROWSER SMOKE ERROR:',error.message);
  console.error('Status:',await page.locator('#stage-status').textContent());
  console.error('Message:',await page.locator('#message').textContent());
  console.error('Other errors:',errors.slice(0,30));
  await page.screenshot({path:'smoke-failed.png',fullPage:true});
  process.exitCode=1;
} finally{await browser.close();}
