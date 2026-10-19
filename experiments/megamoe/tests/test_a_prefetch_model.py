"""Address/credit model, not a proof of native asynchronous DMA correctness."""
def test_a_prefetch_window_and_credit_coverage():
    for ka,kb in ((256,256),(512,512),(512,256)):
        for total in (1,63,64,192,256,320,512,513,3072,7168):
            loaded=[]
            released=[]
            free=[True]*3
            current=0
            for ko in range(0,total,kb):
                if ko==0:
                    assert free[current]
                    free[current]=False
                    loaded.append((ko,min(ka,total-ko),current))
                assert not free[current]
                next_k=ko+min(kb,total-ko)
                if next_k<total and next_k%ka==0:
                    slot=(current+1)%3
                    assert free[slot]  # prior owner's last read must have returned credit
                    free[slot]=False
                    loaded.append((next_k,min(ka,total-next_k),slot))
                if next_k==total or next_k%ka==0:
                    free[current]=True
                    released.append(current)
                    current=(current+1)%3
            assert [(k,n) for k,n,_ in loaded]==[(k,min(ka,total-k)) for k in range(0,total,ka)]
            assert released==[slot for _,_,slot in loaded]
            assert all(free)
