"""Host model of chunk addressing/numerical order; NPU accuracy remains required."""
import numpy as np
import pytest


@pytest.mark.parametrize('hidden,topk,prefix', [(7168,6,258048),(7168,6,129024),(1024,8,16384),(128,1,768)])
def test_packed_chunks_match_weighted_reference(hidden, topk, prefix):
    rng = np.random.default_rng(12)
    # Exactly representable BF16 values, with positive and negative weights.
    inputs = rng.integers(-128,128,(3,topk,hidden)).astype(np.float32)/16
    weights = rng.uniform(-1,1,(3,topk)).astype(np.float32)
    expected = inputs[:,0]*weights[:,0,None]
    for k in range(1,topk):
        expected += inputs[:,k]*weights[:,k,None]
    chunk = min(hidden, (prefix//(4*topk)//128)*128)
    assert chunk > 0 and 4*topk*chunk <= prefix
    actual = np.empty_like(expected)
    covered = np.zeros_like(expected,dtype=np.int32)
    sequence=0
    for token in range(3):
        for channel in range(0,hidden,chunk):
            columns=min(chunk,hidden-channel)
            slot=sequence%2
            sequence+=1
            assert (slot*topk*chunk + topk*columns)*2 <= prefix
            # DMA srcStride is bytes after the copied row, not full H bytes.
            gathered=np.stack([inputs.reshape(-1)[(token*topk+k)*hidden+channel:
                                                       (token*topk+k)*hidden+channel+columns]
                               for k in range(topk)])
            output=gathered[0]*weights[token,0]
            for k in range(1,topk):
                output=np.add(output,np.multiply(gathered[k],weights[token,k]))
            actual[token,channel:channel+columns]=output
            covered[token,channel:channel+columns]+=1
    np.testing.assert_array_equal(covered,1)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
