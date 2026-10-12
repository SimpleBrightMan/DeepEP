"""Compile the actual address transform with a CANN-enum stand-in; hardware test still required."""
from pathlib import Path
import os
import shutil
import subprocess


def test_actual_reused_activation_address(tmp_path):
    compiler = os.environ.get('CXX') or shutil.which('g++')
    assert compiler, 'C++ compiler required for actual address-transform test'
    header = Path(__file__).parents[1] / 'csrc/kernels/mega_moe/catlass_gmm/mega_moe_grouped_matmul.hpp'
    text = header.read_text()
    begin = text.index('constexpr uint64_t ReusedActivationReadAddress(')
    end = text.index('\n}', begin) + 2
    source = '''#include <cstdint>
enum class asc_load_l2_cache_mode : uint8_t { NORMAL_LAST_VICTIM=1 };
''' + text[begin:end] + '''
int main() {
  constexpr uint64_t mask=(uint64_t{1}<<60)-1;
  for (uint64_t old=0; old<16; ++old) {
    for (uint64_t offset : {uint64_t{0}, uint64_t{32}, uint64_t{65536}, mask-65536}) {
      auto result=ReusedActivationReadAddress((old<<60)|offset);
      if ((result & mask)!=offset || (result>>60)!=1) return 1;
      if (ReusedActivationReadAddress(result)!=result) return 2;
    }
  }
  return 0;
}
'''
    source = '#include <initializer_list>\n' + source
    cpp = tmp_path / 'cache.cpp'
    cpp.write_text(source)
    exe = tmp_path / ('cache.exe' if os.name == 'nt' else 'cache')
    subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
