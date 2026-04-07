L0:
(W)     mov (8|M0)               r3.0<1>:ud    r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {@1}
(W)     mul (1|M0)               acc0.0<1>:d   r9.0<0;1,0>:d     r3.2<0;1,0>:uw   {Compacted,@1}
(W)     mach (1|M0)              r4.0<1>:d     r9.0<0;1,0>:d     r3.1<0;1,0>:d    {Compacted}
        add (16|M0)              r5.0<1>:d     r4.0<0;1,0>:d     r1.0<16;16,1>:uw {@1}
        add (16|M16)             r10.0<1>:d    r4.0<0;1,0>:d     r2.0<16;16,1>:uw
        add (16|M0)              r12.0<1>:d    r5.0<8;8,1>:d     r7.0<0;1,0>:d    {Compacted,@2}
        add (16|M16)             r14.0<1>:d    r10.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@2}
        cmp (16|M0)   (lt)f1.0   null<1>:d     r12.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r14.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f1.0)  if (32|M0)                           L672                  L672                
L160:
        shl (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    2:w               {Compacted}
        shl (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    2:w               {Compacted}
        send.dc1 (16|M0)         r20      r16     null    0x0            0x04205E00           {@2,$0} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r22      r18     null    0x0            0x04205E00           {@1,$1} // wr:2+0, rd:2; untyped surface read with x
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r8.7<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r8.7<0;1,0>:d     0:w              
        mul (16|M0)              r24.0<1>:f    r20.0<8;8,1>:f    1.125:f               {Compacted,$0.dst}
        mul (16|M16)             r26.0<1>:f    r22.0<8;8,1>:f    1.125:f               {Compacted,$1.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(~f0.0) if (32|M0)                           L544                  L624                
L280:
        send.dc1 (16|M0)         r40      r16     null    0x0            0x04205E01           {$2} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r42      r18     null    0x0            0x04205E01           {$3} // wr:2+0, rd:2; untyped surface read with x
        mul (16|M0)              acc0.0<1>:f   r24.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r30.0<1>:f    r26.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r34.0<1>:f    r30.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r36.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r38.0<1>:f    r34.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
        mul (16|M0)              acc0.0<1>:f   r40.0<8;8,1>:f    0.25:f               {Compacted,$2.dst}
        mul (16|M16)             r42.0<1>:f    r42.0<8;8,1>:f    0.25:f               {Compacted,$3.dst}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r42.0<1>:f    r42.0<8;8,1>:f    1024.0:f               {Compacted,@2}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r46.0<1>:f    r42.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r48.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r50.0<1>:f    r46.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
        add (16|M0)              acc0.0<1>:f   r36.0<8;8,1>:f    r48.0<8;8,1>:f   {Compacted,@2}
        add (16|M16)             r50.0<1>:f    r38.0<8;8,1>:f    r50.0<8;8,1>:f   {Compacted,@2}
        add (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   -0.03125:f               {Compacted}
        add (16|M16)             r50.0<1>:f    r50.0<8;8,1>:f    -0.03125:f               {Compacted,@2}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r50.0<1>:f    r50.0<8;8,1>:f    1024.0:f               {Compacted,@2}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r54.0<1>:f    r50.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r56.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r58.0<1>:f    r54.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
        else (32|M0)                         L624                  L624                
L544:
        send.dc1 (16|M0)         r60      r16     null    0x0            0x04205E01           {$4} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r62      r18     null    0x0            0x04205E01           {$5} // wr:2+0, rd:2; untyped surface read with x
        mul (16|M0)              r60.0<1>:f    r60.0<8;8,1>:f    0.25:f               {Compacted,$4.dst}
        mul (16|M16)             r62.0<1>:f    r62.0<8;8,1>:f    0.25:f               {Compacted,$5.dst}
        add (16|M0)              acc0.0<1>:f   r24.0<8;8,1>:f    r60.0<8;8,1>:f   {Compacted,@2}
        add (16|M16)             r62.0<1>:f    r26.0<8;8,1>:f    r62.0<8;8,1>:f   {Compacted,@2}
        add (16|M0)              r56.0<1>:f    acc0.0<8;8,1>:f   -0.03125:f               {Compacted}
        add (16|M16)             r58.0<1>:f    r62.0<8;8,1>:f    -0.03125:f               {Compacted,@2}
L624:
        endif (32|M0)                        L640                                
L640:
        send.dc1 (16|M0)         null     r16     r56     0x80            0x04025E02           {@3,$6} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r18     r58     0x80            0x04025E02           {@2,$7} // wr:2+2, rd:0; untyped surface write with x
L672:
        endif (32|M0)                        L688                                
L688:
(W)     mov (8|M0)               r127.0<1>:f   r3.0<8;8,1>:f                    {Compacted}
(W)     send.dc0 (8|M0)          r64      r3      null    0x0            0x0219E000           {$8} // wr:1h+0, rd:1; synchronized global fence flushing
(W)     mov (8|M0)               null<1>:ud    r64.0<8;8,1>:ud                  {$8.dst}
(W)     mov (16|M0)              acc0.0<1>:f   0.0:f                              
(W)     send.ts (1|M0)           null     r127    null    0x0            0x02000010           {EOT,@1} // wr:1+0, rd:0; end of thread
L760:
        nop                    
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
