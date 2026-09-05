import asyncio,unittest
from reload_coordinator import reload_paused_pool
class PoolTests(unittest.IsolatedAsyncioTestCase):
 async def scenario(self,bad=None,version='7'):
  state={'active':0,'peak':0,'resumed':0,'reads':0}
  class Client:
   async def update_weights_from_disk(self,*args,**kwargs):
    state['active']+=1;state['peak']=max(state['peak'],state['active'])
    await asyncio.sleep(.01);state['active']-=1
    return {'success':bad is None}
   async def get_weight_version(self):state['reads']+=1;return version
   async def continue_generation(self):
    assert state['reads']==4
    state['resumed']+=1
  clients=[Client() for _ in range(4)]
  if bad or version!='7':
   with self.assertRaises(RuntimeError):await reload_paused_pool(clients,'/mock','7')
   self.assertEqual(state['resumed'],0)
  else:
   await reload_paused_pool(clients,'/mock','7')
   self.assertEqual(state['resumed'],4)
  self.assertEqual(state['peak'],1)
 async def test_serial_reload_then_all_version_reads_before_resume(self):await self.scenario()
 async def test_reload_failure_keeps_all_paused(self):await self.scenario(bad=True)
 async def test_stale_version_keeps_all_paused(self):await self.scenario(version='6')
if __name__=='__main__':unittest.main()
