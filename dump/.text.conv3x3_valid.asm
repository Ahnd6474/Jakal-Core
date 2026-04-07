L0:
(W)     mov (8|M0)               r100.0<1>:ud  r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {@1}
(W)     mul (1|M0)               acc0.0<1>:d   r9.0<0;1,0>:d     r100.2<0;1,0>:uw {Compacted,@1}
(W)     mach (1|M0)              r5.0<1>:d     r9.0<0;1,0>:d     r100.1<0;1,0>:d  {Compacted}
(W)     mul (1|M0)               acc0.0<1>:d   r9.1<0;1,0>:d     r100.12<0;1,0>:uw
(W)     mach (1|M0)              r14.0<1>:d    r9.1<0;1,0>:d     r100.6<0;1,0>:d 
        add (16|M16)             r12.0<1>:d    r5.0<0;1,0>:d     r2.0<16;16,1>:uw {@3}
        add (16|M16)             r17.0<1>:d    r14.0<0;1,0>:d    r4.0<16;16,1>:uw {@2}
        add (16|M0)              r15.0<1>:d    r14.0<0;1,0>:d    r3.0<16;16,1>:uw
        add (16|M16)             r103.0<1>:d   r12.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r107.0<1>:d   r17.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@3}
(W)     add (1|M0)               r109.0<1>:d   r8.5<0;1,0>:d     -2:w               {Compacted}
(W)     add (1|M0)               r19.0<1>:d    r8.4<0;1,0>:d     -2:w               {Compacted}
        add (16|M0)              r105.0<1>:d   r15.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@5}
        add (16|M0)              r10.0<1>:d    r5.0<0;1,0>:d     r1.0<16;16,1>:uw
        cmp (16|M16)  (lt)f0.0   null<1>:d     r103.0<8;8,1>:ud  r109.0<0;1,0>:ud {@4}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r107.0<8;8,1>:ud  r19.0<0;1,0>:ud  {@4}
(W)     mov (1|M0)               r20.0<1>:hf   0x1:hf                             
        cmp (16|M0)   (lt)f0.0   null<1>:d     r105.0<8;8,1>:ud  r19.0<0;1,0>:ud  {@5}
        add (16|M0)              r101.0<1>:d   r10.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@5}
(f0.0)  sel (16|M16)             acc0.0<1>:uw  r20.0<0;1,0>:uw   0x0:uw              {@3}
(f1.0)  sel (16|M16)             r22.0<1>:uw   r20.0<0;1,0>:uw   0x0:uw             
(f0.0)  cmp (16|M0)   (lt)f0.0   null<1>:d     r101.0<8;8,1>:ud  r109.0<0;1,0>:ud {@3}
        and (16|M16)  (ne)f0.0   null<1>:uw    acc0.0<16;16,1>:uw  r22.0<16;16,1>:uw {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f0.0)  if (32|M0)                           L3984                  L3984                
L352:
(W)     mul (8|M0)               acc0.0<1>:d   r105.0<8;8,1>:d   r8.10<0;1,0>:uw  {Compacted}
        mach (8|M0)              r110.0<1>:d   r105.0<8;8,1>:d   r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r106.0<8;8,1>:d   r8.10<0;1,0>:uw 
        mach (8|M8)              r111.0<1>:d   r106.0<8;8,1>:d   r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r107.0<8;8,1>:d   r8.10<0;1,0>:uw 
        mach (8|M16)             r112.0<1>:d   r107.0<8;8,1>:d   r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M24)              acc0.0<1>:d   r108.0<8;8,1>:d   r8.10<0;1,0>:uw 
        mach (8|M24)             r113.0<1>:d   r108.0<8;8,1>:d   r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r2.0<1>:d     r110.0<8;8,1>:d   r101.0<8;8,1>:d  {Compacted,@5}
        add (16|M16)             r4.0<1>:d     r112.0<8;8,1>:d   r103.0<8;8,1>:d  {Compacted,@2}
        shl (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     2:w               {Compacted,@2}
        shl (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     2:w               {Compacted,@2}
        send.dc1 (16|M0)         r114     r2      null    0x0            0x04205E00           {@2,$0} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r116     r4      null    0x0            0x04205E00           {@1,$1} // wr:2+0, rd:2; untyped surface read with x
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r8.6<0;1,0>:d     0:w              
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r8.6<0;1,0>:d     0:w              
(W)     mov (1|M0)               r109.1<1>:f   0.0625:f                               {Compacted}
(W)     mov (1|M0)               r109.2<1>:f   0.125:f                               {Compacted}
(W)     mov (1|M0)               r109.3<1>:f   0.25:f                               {Compacted}
(W)     mov (1|M0)               r109.4<1>:f   0.0:f                               {Compacted}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) if (32|M0)                           L2672                  L3736                
L600:
        add (16|M0)              r2.0<1>:d     r101.0<8;8,1>:d   1:w               {Compacted,$0.src}
        add (16|M16)             r4.0<1>:d     r103.0<8;8,1>:d   1:w               {Compacted,$1.src}
        add (16|M0)              r6.0<1>:d     r110.0<8;8,1>:d   r2.0<8;8,1>:d    {Compacted,@2}
        add (16|M16)             r9.0<1>:d     r112.0<8;8,1>:d   r4.0<8;8,1>:d    {Compacted,@2}
        shl (16|M0)              r6.0<1>:d     r6.0<8;8,1>:d     2:w               {Compacted,@2}
        shl (16|M16)             r9.0<1>:d     r9.0<8;8,1>:d     2:w               {Compacted,@2}
        send.dc1 (16|M0)         r11      r6      null    0x0            0x04205E00           {@2,$2} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r13      r9      null    0x0            0x04205E00           {@1,$3} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r15.0<1>:d    r101.0<8;8,1>:d   2:w               {Compacted}
        add (16|M16)             r17.0<1>:d    r103.0<8;8,1>:d   2:w               {Compacted}
(W)     mov (1|M0)               r109.5<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
        add (16|M0)              r21.0<1>:d    r110.0<8;8,1>:d   r15.0<8;8,1>:d   {Compacted,@5}
        add (16|M16)             r23.0<1>:d    r112.0<8;8,1>:d   r17.0<8;8,1>:d   {Compacted,@5}
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
        add (16|M0)              r19.0<1>:d    r105.0<8;8,1>:d   1:w               {Compacted}
        shl (16|M0)              r21.0<1>:d    r21.0<8;8,1>:d    2:w               {Compacted,@4}
        shl (16|M16)             r23.0<1>:d    r23.0<8;8,1>:d    2:w               {Compacted,@4}
        add (16|M16)             r25.0<1>:d    r107.0<8;8,1>:d   1:w               {Compacted}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r19.0<8;8,1>:d    r8.10<0;1,0>:uw  {@4}
        send.dc1 (16|M0)         r29      r21     null    0x0            0x04205E00           {@4,$4} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r31      r23     null    0x0            0x04205E00           {@3,$5} // wr:2+0, rd:2; untyped surface read with x
        mach (8|M0)              r27.0<1>:d    r19.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r20.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M8)              r28.0<1>:d    r20.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r25.0<8;8,1>:d    r8.10<0;1,0>:uw  {@5}
        mach (8|M16)             r33.0<1>:d    r25.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r26.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M24)             r34.0<1>:d    r26.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r35.0<1>:d    r27.0<8;8,1>:d    r101.0<8;8,1>:d  {Compacted,@5}
        add (16|M16)             r37.0<1>:d    r33.0<8;8,1>:d    r103.0<8;8,1>:d  {Compacted,@2}
        shl (16|M0)              r35.0<1>:d    r35.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r37.0<1>:d    r37.0<8;8,1>:d    2:w               {Compacted,@2}
        send.dc1 (16|M0)         r39      r35     null    0x0            0x04205E00           {@2,$6} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r41      r37     null    0x0            0x04205E00           {@1,$7} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r47.0<1>:d    r27.0<8;8,1>:d    r2.0<8;8,1>:d    {Compacted}
        add (16|M16)             r49.0<1>:d    r33.0<8;8,1>:d    r4.0<8;8,1>:d    {Compacted}
        mul (16|M0)              acc0.0<1>:f   r114.0<8;8,1>:f   1024.0:f               {Compacted,$0.dst}
        mul (16|M16)             r45.0<1>:f    r116.0<8;8,1>:f   1024.0:f               {Compacted,$1.dst}
        shl (16|M0)              r47.0<1>:d    r47.0<8;8,1>:d    2:w               {Compacted,@4}
        shl (16|M16)             r49.0<1>:d    r49.0<8;8,1>:d    2:w               {Compacted,@4}
        send.dc1 (16|M0)         r60      r47     null    0x0            0x04205E00           {@2,$8} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r62      r49     null    0x0            0x04205E00           {@1,$9} // wr:2+0, rd:2; untyped surface read with x
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r53.0<1>:f    r45.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r58.0<1>:f    r53.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
        add (16|M0)              r70.0<1>:d    r27.0<8;8,1>:d    r15.0<8;8,1>:d   {Compacted}
        add (16|M16)             r72.0<1>:d    r33.0<8;8,1>:d    r17.0<8;8,1>:d   {Compacted}
        mad (16|M0)              acc0.0<1>:f   r109.4<0;0>:f     acc0.0<8;1>:f     r109.1<0>:f     
        mad (16|M16)             r66.0<1>:f    r109.4<0;0>:f     r58.0<8;1>:f      r109.1<0>:f      {@4}
        add (16|M0)              r68.0<1>:d    r105.0<8;8,1>:d   2:w               {Compacted}
        mul (16|M0)              r11.0<1>:f    r11.0<8;8,1>:f    1024.0:f               {Compacted,$2.dst}
        mul (16|M16)             r13.0<1>:f    r13.0<8;8,1>:f    1024.0:f               {Compacted,$3.dst}
        shl (16|M0)              r70.0<1>:d    r70.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r72.0<1>:d    r72.0<8;8,1>:d    2:w               {Compacted,@7}
        mul (16|M0)              r64.0<1>:f    acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r66.0<1>:f    r66.0<8;8,1>:f    1024.0:f               {Compacted,@7}
        add (16|M16)             r82.0<1>:d    r107.0<8;8,1>:d   2:w               {Compacted}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r68.0<8;8,1>:d    r8.10<0;1,0>:uw  {@7}
        send.dc1 (16|M0)         r96      r70     null    0x0            0x04205E00           {@6,$10} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r98      r72     null    0x0            0x04205E00           {@5,$11} // wr:2+0, rd:2; untyped surface read with x
        rnde (16|M0)             r78.0<1>:f    r11.0<8;8,1>:f                   {Compacted,@7}
        rnde (16|M16)            r80.0<1>:f    r13.0<8;8,1>:f                   {Compacted,@7}
        rnde (16|M0)             r74.0<1>:f    r64.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M16)            r76.0<1>:f    r66.0<8;8,1>:f                   {Compacted,@6}
        mach (8|M0)              r84.0<1>:d    r68.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r69.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M8)              r85.0<1>:d    r69.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r82.0<8;8,1>:d    r8.10<0;1,0>:uw  {@7}
        mach (8|M16)             r6.0<1>:d     r82.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r83.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mul (16|M0)              r92.0<1>:f    r78.0<8;8,1>:f    9.765625e-04:f               {Compacted,@7}
        mul (16|M16)             r94.0<1>:f    r80.0<8;8,1>:f    9.765625e-04:f               {Compacted,@7}
        mul (16|M0)              r87.0<1>:f    r74.0<8;8,1>:f    9.765625e-04:f               {Compacted,@7}
        mul (16|M16)             r89.0<1>:f    r76.0<8;8,1>:f    9.765625e-04:f               {Compacted,@7}
        mach (8|M24)             r7.0<1>:d     r83.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r13.0<1>:d    r84.0<8;8,1>:d    r101.0<8;8,1>:d  {Compacted,@7}
        add (16|M16)             r19.0<1>:d    r6.0<8;8,1>:d     r103.0<8;8,1>:d  {Compacted,@2}
        mad (16|M0)              acc0.0<1>:f   r87.0<8;1>:f      r92.0<8;1>:f      r109.2<0>:f      {Compacted,@5}
        mad (16|M16)             r11.0<1>:f    r89.0<8;1>:f      r94.0<8;1>:f      r109.2<0>:f      {Compacted,@5}
        mul (16|M0)              r29.0<1>:f    r29.0<8;8,1>:f    1024.0:f               {Compacted,$4.dst}
        mul (16|M16)             r31.0<1>:f    r31.0<8;8,1>:f    1024.0:f               {Compacted,$5.dst}
        shl (16|M0)              r13.0<1>:d    r13.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r19.0<1>:d    r19.0<8;8,1>:d    2:w               {Compacted,@6}
        mul (16|M0)              r9.0<1>:f     acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r11.0<1>:f    r11.0<8;8,1>:f    1024.0:f               {Compacted,@6}
        send.dc1 (16|M0)         r43      r13     null    0x0            0x04205E00           {@4,$12} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r45      r19     null    0x0            0x04205E00           {@3,$13} // wr:2+0, rd:2; untyped surface read with x
        rnde (16|M0)             acc0.0<1>:f   r29.0<8;8,1>:f                   {@6}
        rnde (16|M16)            r27.0<1>:f    r31.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M0)             r21.0<1>:f    r9.0<8;8,1>:f                    {Compacted,@4}
        rnde (16|M16)            r23.0<1>:f    r11.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r34.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted,$6.src}
        sync.nop                             null                             {Compacted,$7.src}
        mul (16|M16)             r36.0<1>:f    r27.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   r21.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r31.0<1>:f    r23.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        add (16|M0)              r51.0<1>:d    r84.0<8;8,1>:d    r2.0<8;8,1>:d    {Compacted}
        add (16|M16)             r53.0<1>:d    r6.0<8;8,1>:d     r4.0<8;8,1>:d    {Compacted}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r34.0<8;1>:f      r109.1<0>:f      {@6}
        sync.nop                             null                             {Compacted,$9.src}
        mad (16|M16)             r49.0<1>:f    r31.0<8;1>:f      r36.0<8;1>:f      r109.1<0>:f      {Compacted,@4}
        mul (16|M0)              r39.0<1>:f    r39.0<8;8,1>:f    1024.0:f               {Compacted,$6.dst}
        mul (16|M16)             r41.0<1>:f    r41.0<8;8,1>:f    1024.0:f               {Compacted,$7.dst}
        shl (16|M0)              r51.0<1>:d    r51.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r53.0<1>:d    r53.0<8;8,1>:d    2:w               {Compacted,@6}
        mul (16|M0)              r47.0<1>:f    acc0.0<8;8,1>:f   1024.0:f               {Compacted,$8.src}
        mul (16|M16)             r49.0<1>:f    r49.0<8;8,1>:f    1024.0:f               {Compacted,@6}
        send.dc1 (16|M0)         r78      r51     null    0x0            0x04205E00           {@4,$14} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r80      r53     null    0x0            0x04205E00           {@3,$15} // wr:2+0, rd:2; untyped surface read with x
        rnde (16|M0)             acc0.0<1>:f   r39.0<8;8,1>:f                   {@6}
        rnde (16|M16)            r66.0<1>:f    r41.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M0)             r55.0<1>:f    r47.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r57.0<1>:f    r49.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r74.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r76.0<1>:f    r66.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   r55.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        sync.allrd                           ($10,$11)                
        mul (16|M16)             r71.0<1>:f    r57.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        add (16|M0)              r88.0<1>:d    r84.0<8;8,1>:d    r15.0<8;8,1>:d   {Compacted}
        add (16|M16)             r90.0<1>:d    r6.0<8;8,1>:d     r17.0<8;8,1>:d   {Compacted}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r74.0<8;1>:f      r109.2<0>:f      {@6}
        mad (16|M16)             r86.0<1>:f    r71.0<8;1>:f      r76.0<8;1>:f      r109.2<0>:f      {Compacted,@4}
        mul (16|M0)              r60.0<1>:f    r60.0<8;8,1>:f    1024.0:f               {Compacted,$8.dst}
        mul (16|M16)             r62.0<1>:f    r62.0<8;8,1>:f    1024.0:f               {Compacted,$9.dst}
        shl (16|M0)              r88.0<1>:d    r88.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r90.0<1>:d    r90.0<8;8,1>:d    2:w               {Compacted,@6}
        mul (16|M0)              r82.0<1>:f    acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r86.0<1>:f    r86.0<8;8,1>:f    1024.0:f               {Compacted,@6}
        sync.nop                             null                             {Compacted,$13.src}
        send.dc1 (16|M0)         r18      r88     null    0x0            0x04205E00           {@4,$2} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r20      r90     null    0x0            0x04205E00           {@3,$3} // wr:2+0, rd:2; untyped surface read with x
        rnde (16|M0)             acc0.0<1>:f   r60.0<8;8,1>:f                   {@6}
        rnde (16|M16)            r4.0<1>:f     r62.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M0)             r92.0<1>:f    r82.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r94.0<1>:f    r86.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r14.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted,$12.src}
        mul (16|M16)             r16.0<1>:f    r4.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   r92.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r11.0<1>:f    r94.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r14.0<8;1>:f      r109.3<0>:f      {@4}
        mad (16|M16)             r24.0<1>:f    r11.0<8;1>:f      r16.0<8;1>:f      r109.3<0>:f      {Compacted,@2}
        mul (16|M0)              r96.0<1>:f    r96.0<8;8,1>:f    1024.0:f               {Compacted,$10.dst}
        mul (16|M16)             r98.0<1>:f    r98.0<8;8,1>:f    1024.0:f               {Compacted,$11.dst}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r24.0<1>:f    r24.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             r30.0<1>:f    r96.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r32.0<1>:f    r98.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r28.0<1>:f    r24.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r40.0<1>:f    r30.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r48.0<1>:f    r32.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r37.0<1>:f    r28.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r40.0<8;1>:f      r109.2<0>:f      {@4}
        sync.allrd                           ($14,$15)                
        mad (16|M16)             r52.0<1>:f    r37.0<8;1>:f      r48.0<8;1>:f      r109.2<0>:f      {Compacted,@2}
        mul (16|M0)              r43.0<1>:f    r43.0<8;8,1>:f    1024.0:f               {Compacted,$12.dst}
        mul (16|M16)             r45.0<1>:f    r45.0<8;8,1>:f    1024.0:f               {Compacted,$13.dst}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r52.0<1>:f    r52.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             r58.0<1>:f    r43.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r60.0<1>:f    r45.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r56.0<1>:f    r52.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r68.0<1>:f    r58.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r70.0<1>:f    r60.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r65.0<1>:f    r56.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r68.0<8;1>:f      r109.1<0>:f      {@4}
        mad (16|M16)             r74.0<1>:f    r65.0<8;1>:f      r70.0<8;1>:f      r109.1<0>:f      {Compacted,@2}
        mul (16|M0)              r78.0<1>:f    r78.0<8;8,1>:f    1024.0:f               {Compacted,$14.dst}
        mul (16|M16)             r80.0<1>:f    r80.0<8;8,1>:f    1024.0:f               {Compacted,$15.dst}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r74.0<1>:f    r74.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             r84.0<1>:f    r78.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r86.0<1>:f    r80.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r82.0<1>:f    r74.0<8;8,1>:f                   {Compacted,@4}
        mul (16|M0)              r94.0<1>:f    r84.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r96.0<1>:f    r86.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        sync.nop                             null                             {Compacted,$3.src}
        mul (16|M16)             r91.0<1>:f    r82.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r94.0<8;1>:f      r109.2<0>:f      {@4}
        mad (16|M16)             r2.0<1>:f     r91.0<8;1>:f      r96.0<8;1>:f      r109.2<0>:f      {Compacted,@2}
        mul (16|M0)              r18.0<1>:f    r18.0<8;8,1>:f    1024.0:f               {Compacted,$2.dst}
        mul (16|M16)             r20.0<1>:f    r20.0<8;8,1>:f    1024.0:f               {Compacted,$3.dst}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r2.0<1>:f     r2.0<8;8,1>:f     1024.0:f               {Compacted,@4}
        rnde (16|M0)             r9.0<1>:f     r18.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r11.0<1>:f    r20.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r6.0<1>:f     r2.0<8;8,1>:f                    {Compacted,@4}
        mul (16|M0)              r18.0<1>:f    r9.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r20.0<1>:f    r11.0<8;8,1>:f    9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r15.0<1>:f    r6.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r18.0<8;1>:f      r109.1<0>:f      {@4}
        mad (16|M16)             r24.0<1>:f    r15.0<8;1>:f      r20.0<8;1>:f      r109.1<0>:f      {Compacted,@2}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r24.0<1>:f    r24.0<8;8,1>:f    1024.0:f               {Compacted,@2}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r28.0<1>:f    r24.0<8;8,1>:f                   {Compacted,@2}
(W)     mov (1|M0)               f0.0<1>:ud    r109.5<0;1,0>:ud                
        mul (16|M0)              r118.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r120.0<1>:f   r28.0<8;8,1>:f    9.765625e-04:f               {Compacted,@3}
        else (32|M0)                         L3736                  L3736                
L2672:
        add (16|M0)              r2.0<1>:d     r105.0<8;8,1>:d   1:w               {Compacted,$0.src}
(W)     mov (1|M0)               r109.5<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw   {@3}
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
        add (16|M16)             r4.0<1>:d     r107.0<8;8,1>:d   1:w               {Compacted,$1.src}
        add (16|M0)              r6.0<1>:d     r101.0<8;8,1>:d   1:w               {Compacted}
        add (16|M16)             r9.0<1>:d     r103.0<8;8,1>:d   1:w               {Compacted}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r2.0<8;8,1>:d     r8.10<0;1,0>:uw 
        add (16|M0)              r13.0<1>:d    r101.0<8;8,1>:d   2:w               {Compacted}
        add (16|M16)             r15.0<1>:d    r103.0<8;8,1>:d   2:w               {Compacted}
        mach (8|M0)              r11.0<1>:d    r2.0<8;8,1>:d     r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r3.0<8;8,1>:d     r8.10<0;1,0>:uw 
        add (16|M0)              r17.0<1>:d    r110.0<8;8,1>:d   r6.0<8;8,1>:d    {Compacted,@7}
        add (16|M16)             r19.0<1>:d    r112.0<8;8,1>:d   r9.0<8;8,1>:d    {Compacted,@7}
        mach (8|M8)              r12.0<1>:d    r3.0<8;8,1>:d     r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r4.0<8;8,1>:d     r8.10<0;1,0>:uw  {@7}
        add (16|M0)              r23.0<1>:d    r105.0<8;8,1>:d   2:w               {Compacted}
        mach (8|M16)             r21.0<1>:d    r4.0<8;8,1>:d     r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r5.0<8;8,1>:d     r8.10<0;1,0>:uw 
        add (16|M0)              r25.0<1>:d    r110.0<8;8,1>:d   r13.0<8;8,1>:d   {Compacted,@7}
        add (16|M16)             r27.0<1>:d    r112.0<8;8,1>:d   r15.0<8;8,1>:d   {Compacted,@7}
        mach (8|M24)             r22.0<1>:d    r5.0<8;8,1>:d     r8.5<0;1,0>:d    {Compacted}
        add (16|M16)             r29.0<1>:d    r107.0<8;8,1>:d   2:w               {Compacted}
        shl (16|M0)              r17.0<1>:d    r17.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r19.0<1>:d    r19.0<8;8,1>:d    2:w               {Compacted,@7}
        add (16|M0)              r31.0<1>:d    r11.0<8;8,1>:d    r101.0<8;8,1>:d  {Compacted,@7}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r23.0<8;8,1>:d    r8.10<0;1,0>:uw  {@7}
        add (16|M16)             r33.0<1>:d    r21.0<8;8,1>:d    r103.0<8;8,1>:d  {Compacted,@6}
        shl (16|M0)              r25.0<1>:d    r25.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r27.0<1>:d    r27.0<8;8,1>:d    2:w               {Compacted,@7}
        mach (8|M0)              r40.0<1>:d    r23.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r24.0<8;8,1>:d    r8.10<0;1,0>:uw 
        send.dc1 (16|M0)         r36      r17     null    0x0            0x04205E00           {@7,$4} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r38      r19     null    0x0            0x04205E00           {@7,$5} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r42.0<1>:d    r11.0<8;8,1>:d    r6.0<8;8,1>:d    {Compacted}
        add (16|M16)             r44.0<1>:d    r21.0<8;8,1>:d    r9.0<8;8,1>:d    {Compacted}
        mach (8|M8)              r41.0<1>:d    r24.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r29.0<8;8,1>:d    r8.10<0;1,0>:uw 
        shl (16|M0)              r31.0<1>:d    r31.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r33.0<1>:d    r33.0<8;8,1>:d    2:w               {Compacted,@7}
        mach (8|M16)             r50.0<1>:d    r29.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r30.0<8;8,1>:d    r8.10<0;1,0>:uw 
        send.dc1 (16|M0)         r46      r25     null    0x0            0x04205E00           {$6} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r48      r27     null    0x0            0x04205E00           {@7,$7} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r52.0<1>:d    r11.0<8;8,1>:d    r13.0<8;8,1>:d   {Compacted}
        add (16|M16)             r54.0<1>:d    r21.0<8;8,1>:d    r15.0<8;8,1>:d   {Compacted}
        mach (8|M24)             r51.0<1>:d    r30.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        shl (16|M0)              r42.0<1>:d    r42.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r44.0<1>:d    r44.0<8;8,1>:d    2:w               {Compacted,@7}
        send.dc1 (16|M0)         r56      r31     null    0x0            0x04205E00           {@7,$8} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r58      r33     null    0x0            0x04205E00           {@7,$9} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r60.0<1>:d    r40.0<8;8,1>:d    r101.0<8;8,1>:d  {Compacted,@7}
        add (16|M16)             r62.0<1>:d    r50.0<8;8,1>:d    r103.0<8;8,1>:d  {Compacted,@4}
        shl (16|M0)              r52.0<1>:d    r52.0<8;8,1>:d    2:w               {Compacted,@7}
        shl (16|M16)             r54.0<1>:d    r54.0<8;8,1>:d    2:w               {Compacted,@7}
        send.dc1 (16|M0)         r64      r42     null    0x0            0x04205E00           {@6,$10} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r66      r44     null    0x0            0x04205E00           {@5,$11} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r68.0<1>:d    r40.0<8;8,1>:d    r6.0<8;8,1>:d    {Compacted}
        add (16|M16)             r70.0<1>:d    r50.0<8;8,1>:d    r9.0<8;8,1>:d    {Compacted}
        shl (16|M0)              r60.0<1>:d    r60.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r62.0<1>:d    r62.0<8;8,1>:d    2:w               {Compacted,@6}
        send.dc1 (16|M0)         r72      r52     null    0x0            0x04205E00           {@6,$12} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r74      r54     null    0x0            0x04205E00           {@5,$13} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r76.0<1>:d    r40.0<8;8,1>:d    r13.0<8;8,1>:d   {Compacted}
        add (16|M16)             r78.0<1>:d    r50.0<8;8,1>:d    r15.0<8;8,1>:d   {Compacted}
        shl (16|M0)              r68.0<1>:d    r68.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r70.0<1>:d    r70.0<8;8,1>:d    2:w               {Compacted,@6}
        send.dc1 (16|M0)         r80      r60     null    0x0            0x04205E00           {@6,$14} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r82      r62     null    0x0            0x04205E00           {@5,$15} // wr:2+0, rd:2; untyped surface read with x
        shl (16|M0)              r76.0<1>:d    r76.0<8;8,1>:d    2:w               {Compacted,@4}
        shl (16|M16)             r78.0<1>:d    r78.0<8;8,1>:d    2:w               {Compacted,@4}
        send.dc1 (16|M0)         r84      r68     null    0x0            0x04205E00           {@4,$2} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r86      r70     null    0x0            0x04205E00           {@3,$3} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r88      r76     null    0x0            0x04205E00           {@2,$0} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r90      r78     null    0x0            0x04205E00           {@1,$1} // wr:2+0, rd:2; untyped surface read with x
        mad (16|M0)              acc0.0<1>:f   r109.4<0;0>:f     r114.0<8;1>:f     r109.1<0>:f      {$0.dst}
        mad (16|M16)             r95.0<1>:f    r109.4<0;0>:f     r116.0<8;1>:f     r109.1<0>:f      {$1.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r36.0<8;1>:f      r109.2<0>:f      {$4.dst}
        mad (16|M16)             r2.0<1>:f     r95.0<8;1>:f      r38.0<8;1>:f      r109.2<0>:f      {Compacted,@2,$5.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r46.0<8;1>:f      r109.1<0>:f      {$6.dst}
        mad (16|M16)             r7.0<1>:f     r2.0<8;1>:f       r48.0<8;1>:f      r109.1<0>:f      {Compacted,@2,$7.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r56.0<8;1>:f      r109.2<0>:f      {$8.dst}
        mad (16|M16)             r11.0<1>:f    r7.0<8;1>:f       r58.0<8;1>:f      r109.2<0>:f      {Compacted,@2,$9.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r64.0<8;1>:f      r109.3<0>:f      {$10.dst}
        mad (16|M16)             r15.0<1>:f    r11.0<8;1>:f      r66.0<8;1>:f      r109.3<0>:f      {Compacted,@2,$11.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r72.0<8;1>:f      r109.2<0>:f      {$12.dst}
        mad (16|M16)             r19.0<1>:f    r15.0<8;1>:f      r74.0<8;1>:f      r109.2<0>:f      {Compacted,@2,$13.dst}
(W)     mov (1|M0)               f0.0<1>:ud    r109.5<0;1,0>:ud                
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r80.0<8;1>:f      r109.1<0>:f      {$14.dst}
        mad (16|M16)             r23.0<1>:f    r19.0<8;1>:f      r82.0<8;1>:f      r109.1<0>:f      {Compacted,@3,$15.dst}
        mad (16|M0)              acc0.0<1>:f   acc0.0<8;1>:f     r84.0<8;1>:f      r109.2<0>:f      {$2.dst}
        mad (16|M16)             r27.0<1>:f    r23.0<8;1>:f      r86.0<8;1>:f      r109.2<0>:f      {Compacted,@2,$3.dst}
        mad (16|M0)              r118.0<1>:f   acc0.0<8;1>:f     r88.0<8;1>:f      r109.1<0>:f      {$0.dst}
        mad (16|M16)             r120.0<1>:f   r27.0<8;1>:f      r90.0<8;1>:f      r109.1<0>:f      {Compacted,@2,$1.dst}
L3736:
        endif (32|M0)                        L3752                                
L3752:
(W)     mov (1|M0)               r109.5<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r105.0<8;8,1>:d   r109.0<0;1,0>:uw
        mach (8|M0)              r2.0<1>:d     r105.0<8;8,1>:d   r109.0<0;1,0>:d  {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r106.0<8;8,1>:d   r109.0<0;1,0>:uw
        mach (8|M8)              r3.0<1>:d     r106.0<8;8,1>:d   r109.0<0;1,0>:d  {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r107.0<8;8,1>:d   r109.0<0;1,0>:uw
        mach (8|M16)             r4.0<1>:d     r107.0<8;8,1>:d   r109.0<0;1,0>:d  {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r108.0<8;8,1>:d   r109.0<0;1,0>:uw
        mach (8|M24)             r5.0<1>:d     r108.0<8;8,1>:d   r109.0<0;1,0>:d  {Compacted}
        add (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     r101.0<8;8,1>:d  {Compacted,@5}
        add (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     r103.0<8;8,1>:d  {Compacted,@2}
        shl (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     2:w               {Compacted,@2}
        shl (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     2:w               {Compacted,@2}
        send.dc1 (16|M0)         null     r2      r118    0x80            0x04025E01           {@2,$4} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r4      r120    0x80            0x04025E01           {@1,$5} // wr:2+2, rd:0; untyped surface write with x
(W)     mov (1|M0)               f0.0<1>:ud    r109.5<0;1,0>:ud                
L3984:
        endif (32|M0)                        L4000                                
L4000:
(W)     mov (8|M0)               r127.0<1>:f   r100.0<8;8,1>:f                  {Compacted}
        sync.nop                             null                             {Compacted,$4.src}
(W)     send.dc0 (8|M0)          r2       r100    null    0x0            0x0219E000           {$6} // wr:1h+0, rd:1; synchronized global fence flushing
(W)     mov (8|M0)               null<1>:ud    r2.0<8;8,1>:ud                   {$6.dst}
(W)     mov (16|M0)              acc0.0<1>:f   0.0:f                              
(W)     send.ts (1|M0)           null     r127    null    0x0            0x02000010           {EOT,@1} // wr:1+0, rd:0; end of thread
L4080:
        nop                    
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
