"""缓存参数在读取凭据、连接SSH或创建输出目录前拒绝越界/原地路径。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CacheArguments(unittest.TestCase):
    def test_invalid_cache_rejected_before_io(self):
        for cache in ('/etc', '/home/codex-deepep/new-root', '/home/codex-deepep/../outside', '/home/codex-deepep/x;true'):
            with self.subTest(cache=cache), tempfile.TemporaryDirectory() as temporary:
                target=Path(temporary)/'results'
                command=[sys.executable,str(Path(__file__).with_name('campaign.py')),
                    '--inventory','does-not-exist','--host-keys','does-not-exist',
                    '--dependencies','does-not-exist','--packages','does-not-exist',
                    '--output',str(target),'--rack','01','--nodes','CPU8,CPU6,CPU5,CPU7',
                    '--ep','32','--remote-root','/home/codex-deepep/new-root',
                    '--port-base','24520','--method-only','control','--package-cache-root',cache]
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Invalid distinct package cache root',result.stderr)
                self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
