L0:
(W)     mov (8|M0)               r85.0<1>:ud   r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {@1}
(W)     mul (1|M0)               acc0.0<1>:d   r9.3<0;1,0>:d     r85.2<0;1,0>:uw  {@1}
(W)     mach (1|M0)              r5.0<1>:d     r9.3<0;1,0>:d     r85.1<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r9.4<0;1,0>:d     r85.12<0;1,0>:uw
(W)     mach (1|M0)              r14.0<1>:d    r9.4<0;1,0>:d     r85.6<0;1,0>:d  
        add (16|M16)             r12.0<1>:d    r5.0<0;1,0>:d     r2.0<16;16,1>:uw {@3}
        add (16|M16)             r17.0<1>:d    r14.0<0;1,0>:d    r4.0<16;16,1>:uw {@2}
        add (16|M0)              r15.0<1>:d    r14.0<0;1,0>:d    r3.0<16;16,1>:uw
        add (16|M16)             r88.0<1>:d    r12.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r92.0<1>:d    r17.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@3}
        add (16|M0)              r90.0<1>:d    r15.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@3}
        add (16|M0)              r10.0<1>:d    r5.0<0;1,0>:d     r1.0<16;16,1>:uw
        cmp (16|M16)  (lt)f0.0   null<1>:d     r88.0<8;8,1>:ud   r8.7<0;1,0>:ud   {@4}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r92.0<8;8,1>:ud   r9.1<0;1,0>:ud   {@4}
(W)     mov (1|M0)               r19.0<1>:hf   0x1:hf                             
        cmp (16|M0)   (lt)f0.0   null<1>:d     r90.0<8;8,1>:ud   r9.1<0;1,0>:ud   {@5}
        add (16|M0)              r86.0<1>:d    r10.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@5}
(f0.0)  sel (16|M16)             acc0.0<1>:uw  r19.0<0;1,0>:uw   0x0:uw              {@3}
(f1.0)  sel (16|M16)             r21.0<1>:uw   r19.0<0;1,0>:uw   0x0:uw             
(f0.0)  cmp (16|M0)   (lt)f0.0   null<1>:d     r86.0<8;8,1>:ud   r8.7<0;1,0>:ud   {@3}
        and (16|M16)  (ne)f0.0   null<1>:uw    acc0.0<16;16,1>:uw  r21.0<16;16,1>:uw {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f0.0)  if (32|M0)                           L3464                  L3464                
L352:
(W)     mov (1|M0)               r2.0<1>:f     r8.6<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               r4.0<1>:f     0x4F800000:f                               {Compacted}
(W)     and (1|M0)    (eq)f0.0   r3.0<1>:d     r2.0<0;1,0>:d     2139095040:d               {@2}
(W)     and (1|M0)               r5.0<1>:d     r2.0<0;1,0>:d     8388607:d              
(W)     and (1|M0)               r13.0<1>:d    r2.0<0;1,0>:d     8388607:d              
(W)     mov (1|M0)               r7.0<1>:f     r8.7<0;1,0>:ud                   {Compacted}
(W&f0.0) sel (1|M0)              r6.0<1>:f     r4.0<0;1,0>:f     1.0:f               {@5}
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r5.0<0;1,0>:d     0:w               {Compacted,@4}
(W)     cmp (1|M0)    (ge)f1.0   null<1>:ud    r3.0<0;1,0>:ud    0x64000000:ud              {@6}
        add (16|M0)              r10.0<1>:d    r90.0<8;8,1>:d    r9.0<0;1,0>:d    {Compacted}
(W)     mov (1|M0)               r12.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W&~f1.0) sel (1|M0)             r14.0<1>:f    r6.0<0;1,0>:f     0x2F800000:f               {@5}
(W)     mov (1|M0)               f0.0<1>:ud    r12.0<0;1,0>:ud                  {@2}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r13.0<0;1,0>:d    0:w               {@7}
(W)     and (1|M0)    (eq)f1.0   r15.0<1>:d    r7.0<0;1,0>:d     2139095040:d               {@7}
        mov (16|M0)              acc0.0<1>:f   r10.0<8;8,1>:ud                  {@6}
(W)     mov (1|M0)               r12.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
        add (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   0.5:f               {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r12.0<0;1,0>:ud                  {@2}
(W&f1.0) sel (1|M0)              r28.0<1>:f    r4.0<0;1,0>:f     1.0:f              
(W)     mov (1|M0)               r21.0<1>:f    r8.4<0;1,0>:ud                   {Compacted}
        add (16|M16)             r18.0<1>:d    r92.0<8;8,1>:d    r9.0<0;1,0>:d    {Compacted}
(W&~f0.0) cmp (16|M0) (eq)f0.0   null<1>:d     r3.0<0;1,0>:d     0:w              
        mul (16|M0)              r29.0<1>:f    acc0.0<8;8,1>:f   r21.0<0;1,0>:f   {Compacted,@3}
(W)     mov (1|M0)               r12.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     cmp (1|M0)    (ge)f0.0   null<1>:ud    r15.0<0;1,0>:ud   0x64000000:ud              {@7}
        mov (16|M16)             acc0.0<1>:f   r18.0<8;8,1>:ud                  {@5}
(W)     mul (1|M0)               r20.0<1>:f    r2.0<0;1,0>:f     r14.0<0;1,0>:f   {Compacted}
(W)     mov (1|M0)               r24.0<1>:uw   f0.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r12.0<0;1,0>:ud                  {@5}
        add (16|M16)             acc0.0<1>:f   acc0.0<8;8,1>:f   0.5:f               {Compacted}
(W)     and (1|M0)               r25.0<1>:d    r7.0<0;1,0>:d     8388607:d              
(W&~f0.0) cmp (16|M16) (eq)f0.0  null<1>:d     r3.0<0;1,0>:d     0:w              
(W)     and (1|M0)               r26.0<1>:d    r7.0<0;1,0>:d     8388607:d              
(W)     math.inv (1|M0)          r27.0<1>:f    r20.0<0;1,0>:f                   {@7,$0}
(W)     mov (1|M0)               r12.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:uw    r24.0<0;1,0>:uw                  {@7}
(W)     not (1|M0)               r31.0<1>:uw   r12.0<0;1,0>:uw                  {@2}
(W)     not (1|M16)              r31.1<1>:uw   r12.1<0;1,0>:uw                 
        mul (16|M16)             r35.0<1>:f    acc0.0<8;8,1>:f   r21.0<0;1,0>:f   {Compacted}
(W&~f0.0) sel (1|M0)             r32.0<1>:f    r28.0<0;1,0>:f    0x2F800000:f              
(W)     mov (1|M0)               f0.0<1>:ud    r31.0<0;1,0>:ud                  {@3}
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r25.0<0;1,0>:d    0:w               {@7}
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r26.0<0;1,0>:d    0:w               {@7}
(f0.0)  cmp (16|M0)   (eq)f0.0   null<1>:f     r29.0<8;8,1>:f    r2.0<0;1,0>:f   
        mov (16|M0)              r37.0<1>:f    r86.0<8;8,1>:ud                  {Compacted}
(W)     mov (1|M0)               r31.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
        mov (16|M16)             r39.0<1>:f    r88.0<8;8,1>:ud                  {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r31.0<0;1,0>:ud                  {@2}
        mul (16|M0)              r33.0<1>:f    r27.0<0;1,0>:f    r29.0<8;8,1>:f   {Compacted,$0.dst}
(W)     mul (1|M0)               r41.0<1>:f    r7.0<0;1,0>:f     r32.0<0;1,0>:f   {Compacted,@7}
(f0.0)  cmp (16|M16)  (eq)f0.0   null<1>:f     r35.0<8;8,1>:f    r2.0<0;1,0>:f   
(W&~f1.0) cmp (16|M0) (eq)f1.0   null<1>:d     r15.0<0;1,0>:d    0:w              
(W&~f1.0) cmp (16|M16) (eq)f1.0  null<1>:d     r15.0<0;1,0>:d    0:w              
        add (16|M0)              r37.0<1>:f    r37.0<8;8,1>:f    0.5:f               {Compacted,@7}
        add (16|M16)             r39.0<1>:f    r39.0<8;8,1>:f    0.5:f               {Compacted,@7}
(W)     mov (1|M0)               r46.0<1>:f    r8.5<0;1,0>:ud                   {Compacted}
        mul (16|M0)              acc0.0<1>:f   r33.0<8;8,1>:f    r14.0<0;1,0>:f   {Compacted,@7}
        mul (16|M16)             r44.0<1>:f    r27.0<0;1,0>:f    r35.0<8;8,1>:f   {Compacted}
(W)     math.inv (1|M0)          r47.0<1>:f    r41.0<0;1,0>:f                   {@7,$1}
(W)     mov (1|M0)               r31.0<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
        mul (16|M0)              r52.0<1>:f    r37.0<8;8,1>:f    r46.0<0;1,0>:f   {Compacted,@4}
        mul (16|M16)             r54.0<1>:f    r39.0<8;8,1>:f    r46.0<0;1,0>:f   {Compacted,@6}
(~f0.0) sel (16|M0)              r48.0<1>:f    acc0.0<8;8,1>:f   1.0:f              
(W)     not (1|M16)              f0.1<1>:uw    f1.1<0;1,0>:uw                  
(W)     not (1|M0)               f0.0<1>:uw    f1.0<0;1,0>:uw                   {Compacted}
(W)     mov (1|M0)               f1.0<1>:ud    r31.0<0;1,0>:ud                  {@6}
        mul (16|M16)             acc0.0<1>:f   r44.0<8;8,1>:f    r14.0<0;1,0>:f   {Compacted,@7}
        mul (16|M0)              r59.0<1>:f    r47.0<0;1,0>:f    r52.0<8;8,1>:f   {Compacted,@7,$1.dst}
        mul (16|M16)             r61.0<1>:f    r47.0<0;1,0>:f    r54.0<8;8,1>:f   {Compacted,@7}
        add (16|M0)              r48.0<1>:f    r48.0<8;8,1>:f    -0.5:f               {Compacted,@7}
(f0.0)  cmp (16|M16)  (eq)f0.0   null<1>:f     r54.0<8;8,1>:f    r7.0<0;1,0>:f   
(f0.0)  cmp (16|M0)   (eq)f0.0   null<1>:f     r52.0<8;8,1>:f    r7.0<0;1,0>:f   
(~f1.0) sel (16|M16)             acc0.0<1>:f   acc0.0<8;8,1>:f   1.0:f              
(W)     add (1|M0)               r56.0<1>:d    r8.4<0;1,0>:d     -1:w               {Compacted}
        mul (16|M0)              r66.0<1>:f    r59.0<8;8,1>:f    r32.0<0;1,0>:f   {Compacted,@7}
        mul (16|M16)             r68.0<1>:f    r61.0<8;8,1>:f    r32.0<0;1,0>:f   {Compacted,@7}
        add (16|M16)             r57.0<1>:f    acc0.0<8;8,1>:f   -0.5:f               {Compacted}
        sel (16|M0)   (ge)f0.0   r64.0<1>:f    r48.0<8;8,1>:f    0.0:f               {Compacted,@7}
(W)     mov (1|M0)               r63.0<1>:f    r56.0<0;1,0>:ud                  {Compacted,@5}
(~f0.0) sel (16|M0)              acc0.0<1>:f   r66.0<8;8,1>:f    1.0:f               {@5}
(~f0.0) sel (16|M16)             r76.0<1>:f    r68.0<8;8,1>:f    1.0:f               {@5}
        sel (16|M16)  (ge)f0.0   r72.0<1>:f    r57.0<8;8,1>:f    0.0:f               {@5}
        sel (16|M0)   (lt)f0.0   r70.0<1>:f    r64.0<8;8,1>:f    r63.0<0;1,0>:f   {Compacted,@4}
        add (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   -0.5:f               {Compacted}
        add (16|M16)             r76.0<1>:f    r76.0<8;8,1>:f    -0.5:f               {Compacted,@4}
(W)     add (1|M0)               r82.0<1>:d    r8.5<0;1,0>:d     -1:w               {Compacted}
        sel (16|M16)  (lt)f0.0   r80.0<1>:f    r72.0<8;8,1>:f    r63.0<0;1,0>:f   {@5}
        mov (16|M0)              r78.0<1>:ud   r70.0<8;8,1>:f                   {@5}
        sel (16|M0)   (ge)f0.0   r3.0<1>:f     acc0.0<8;8,1>:f   0.0:f               {Compacted}
        sel (16|M16)  (ge)f0.0   r5.0<1>:f     r76.0<8;8,1>:f    0.0:f               {@5}
(W)     mov (1|M0)               r2.0<1>:f     r82.0<0;1,0>:ud                  {Compacted,@5}
        mov (16|M16)             r83.0<1>:ud   r80.0<8;8,1>:f                   {@5}
(W)     mul (8|M0)               acc0.0<1>:d   r78.0<8;8,1>:d    r8.10<0;1,0>:uw  {Compacted,@5}
        mach (8|M0)              r115.0<1>:d   r78.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r79.0<8;8,1>:d    r8.10<0;1,0>:uw 
        sel (16|M0)   (lt)f0.0   r10.0<1>:f    r3.0<8;8,1>:f     r2.0<0;1,0>:f    {Compacted,@5}
        sel (16|M16)  (lt)f0.0   r12.0<1>:f    r5.0<8;8,1>:f     r2.0<0;1,0>:f    {@7}
        mach (8|M8)              r116.0<1>:d   r79.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r83.0<8;8,1>:d    r8.10<0;1,0>:uw  {@7}
        mach (8|M16)             r117.0<1>:d   r83.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W)     mul (8|M24)              acc0.0<1>:d   r84.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mov (16|M0)              r102.0<1>:ud  r10.0<8;8,1>:f                   {@6}
        mov (16|M16)             r104.0<1>:ud  r12.0<8;8,1>:f                   {@6}
        mach (8|M24)             r118.0<1>:d   r84.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r14.0<1>:d    r115.0<8;8,1>:d   r102.0<8;8,1>:d  {Compacted,@3}
        add (16|M16)             r16.0<1>:d    r117.0<8;8,1>:d   r104.0<8;8,1>:d  {Compacted,@2}
        shl (16|M0)              r14.0<1>:d    r14.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@2}
        send.dc1 (16|M0)         r119     r14     null    0x0            0x04205E00           {@2,$2} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r121     r16     null    0x0            0x04205E00           {@1,$3} // wr:2+0, rd:2; untyped surface read with x
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r9.2<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r9.2<0;1,0>:d     0:w              
        add (16|M0)              r18.0<1>:d    r78.0<8;8,1>:d    1:w               {Compacted}
        add (16|M16)             r20.0<1>:d    r83.0<8;8,1>:d    1:w               {Compacted}
        add (16|M0)              r22.0<1>:d    r102.0<8;8,1>:d   1:w               {Compacted}
        add (16|M16)             r24.0<1>:d    r104.0<8;8,1>:d   1:w               {Compacted}
        mov (16|M0)              r26.0<1>:f    r78.0<8;8,1>:ud                  {Compacted}
        mov (16|M16)             r28.0<1>:f    r83.0<8;8,1>:ud                  {Compacted}
        mov (16|M0)              r30.0<1>:f    r102.0<8;8,1>:ud                 {Compacted}
        mov (16|M16)             r32.0<1>:f    r104.0<8;8,1>:ud                 {Compacted}
        sel (16|M0)   (lt)f0.0   r94.0<1>:ud   r56.0<0;1,0>:ud   r18.0<8;8,1>:ud  {Compacted,@7}
        sel (16|M16)  (lt)f0.0   r96.0<1>:ud   r56.0<0;1,0>:ud   r20.0<8;8,1>:ud  {@7}
        sel (16|M0)   (lt)f0.0   r106.0<1>:ud  r82.0<0;1,0>:ud   r22.0<8;8,1>:ud  {Compacted,@7}
        sel (16|M16)  (lt)f0.0   r108.0<1>:ud  r82.0<0;1,0>:ud   r24.0<8;8,1>:ud  {@7}
        add (16|M0)              r98.0<1>:f    r70.0<8;8,1>:f    -r26.0<8;8,1>:f  {Compacted,@7}
        add (16|M16)             r100.0<1>:f   r80.0<8;8,1>:f    -r28.0<8;8,1>:f  {Compacted,@7}
        add (16|M0)              r111.0<1>:f   r10.0<8;8,1>:f    -r30.0<8;8,1>:f  {Compacted,@7}
        add (16|M16)             r113.0<1>:f   r12.0<8;8,1>:f    -r32.0<8;8,1>:f  {Compacted,@7}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L2712                  L3216                
L1872:
(W)     mov (1|M0)               r8.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
        add (16|M0)              r6.0<1>:d     r115.0<8;8,1>:d   r106.0<8;8,1>:d  {Compacted}
        add (16|M16)             r9.0<1>:d     r117.0<8;8,1>:d   r108.0<8;8,1>:d  {Compacted}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r94.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M0)              r2.0<1>:d     r94.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r95.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M8)              r3.0<1>:d     r95.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r96.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M16)             r4.0<1>:d     r96.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r97.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M24)             r5.0<1>:d     r97.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r11.0<1>:d    r2.0<8;8,1>:d     r102.0<8;8,1>:d  {Compacted,@5}
        sync.nop                             null                             {Compacted,$3.src}
        add (16|M0)              r15.0<1>:d    r2.0<8;8,1>:d     r106.0<8;8,1>:d  {Compacted,$2.src}
        add (16|M16)             r13.0<1>:d    r4.0<8;8,1>:d     r104.0<8;8,1>:d  {Compacted,@3}
        add (16|M16)             r17.0<1>:d    r4.0<8;8,1>:d     r108.0<8;8,1>:d  {Compacted}
        shl (16|M0)              r6.0<1>:d     r6.0<8;8,1>:d     2:w               {Compacted}
        shl (16|M16)             r9.0<1>:d     r9.0<8;8,1>:d     2:w               {Compacted}
        shl (16|M0)              r11.0<1>:d    r11.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M0)              r15.0<1>:d    r15.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r13.0<1>:d    r13.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r17.0<1>:d    r17.0<8;8,1>:d    2:w               {Compacted,@6}
        send.dc1 (16|M0)         r19      r6      null    0x0            0x04205E00           {@6,$4} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r21      r9      null    0x0            0x04205E00           {@5,$5} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r23      r11     null    0x0            0x04205E00           {@4,$6} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r27      r15     null    0x0            0x04205E00           {@3,$7} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r25      r13     null    0x0            0x04205E00           {@2,$8} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r29      r17     null    0x0            0x04205E00           {@1,$9} // wr:2+0, rd:2; untyped surface read with x
        mul (16|M0)              acc0.0<1>:f   r119.0<8;8,1>:f   1024.0:f               {Compacted,$2.dst}
        mul (16|M16)             r33.0<1>:f    r121.0<8;8,1>:f   1024.0:f               {Compacted,$3.dst}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r37.0<1>:f    r33.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r51.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r53.0<1>:f    r37.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
(W)     mov (1|M0)               f0.0<1>:ud    r8.0<0;1,0>:ud                  
        mul (16|M0)              acc0.0<1>:f   r19.0<8;8,1>:f    1024.0:f               {Compacted,$4.dst}
        mul (16|M16)             r21.0<1>:f    r21.0<8;8,1>:f    1024.0:f               {Compacted,$5.dst}
        mul (16|M0)              r23.0<1>:f    r23.0<8;8,1>:f    1024.0:f               {Compacted,$6.dst}
        mul (16|M0)              r27.0<1>:f    r27.0<8;8,1>:f    1024.0:f               {Compacted,$7.dst}
        mul (16|M16)             r25.0<1>:f    r25.0<8;8,1>:f    1024.0:f               {Compacted,$8.dst}
        mul (16|M16)             r29.0<1>:f    r29.0<8;8,1>:f    1024.0:f               {Compacted,$9.dst}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r41.0<1>:f    r21.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M0)             r43.0<1>:f    r23.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M0)             r47.0<1>:f    r27.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M16)            r45.0<1>:f    r25.0<8;8,1>:f                   {Compacted,@6}
        rnde (16|M16)            r49.0<1>:f    r29.0<8;8,1>:f                   {Compacted,@6}
        mul (16|M0)              r55.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r57.0<1>:f    r41.0<8;8,1>:f    9.765625e-04:f               {Compacted,@6}
        mul (16|M0)              r59.0<1>:f    r43.0<8;8,1>:f    9.765625e-04:f               {Compacted,@6}
        mul (16|M0)              r63.0<1>:f    r47.0<8;8,1>:f    9.765625e-04:f               {Compacted,@6}
        mul (16|M16)             r61.0<1>:f    r45.0<8;8,1>:f    9.765625e-04:f               {Compacted,@6}
        mul (16|M16)             acc0.0<1>:f   r49.0<8;8,1>:f    9.765625e-04:f               {Compacted,@6}
        add (16|M0)              r68.0<1>:f    r55.0<8;8,1>:f    -r51.0<8;8,1>:f  {Compacted,@6}
        add (16|M16)             r70.0<1>:f    r57.0<8;8,1>:f    -r53.0<8;8,1>:f  {Compacted,@6}
        add (16|M0)              r72.0<1>:f    r63.0<8;8,1>:f    -r59.0<8;8,1>:f  {Compacted,@5}
        add (16|M16)             acc0.0<1>:f   acc0.0<8;8,1>:f   -r61.0<8;8,1>:f  {Compacted,@5}
        mad (16|M0)              r76.0<1>:f    r51.0<8;1>:f      r68.0<8;1>:f      r111.0<1>:f      {Compacted,@4}
        mad (16|M16)             r78.0<1>:f    r53.0<8;1>:f      r70.0<8;1>:f      r113.0<1>:f      {Compacted,@4}
        mad (16|M0)              r80.0<1>:f    r59.0<8;1>:f      r72.0<8;1>:f      r111.0<1>:f      {Compacted,@4}
        mad (16|M16)             acc0.0<1>:f   r61.0<8;1>:f      acc0.0<8;1>:f     r113.0<1>:f      {Compacted}
        mul (16|M0)              r76.0<1>:f    r76.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r78.0<1>:f    r78.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M0)              r80.0<1>:f    r80.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        rnde (16|M0)             r2.0<1>:f     r76.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r4.0<1>:f     r78.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M0)             r6.0<1>:f     r80.0<8;8,1>:f                   {Compacted,@4}
        rnde (16|M16)            r9.0<1>:f     acc0.0<8;8,1>:f                  {Compacted}
        mul (16|M0)              r11.0<1>:f    r2.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r13.0<1>:f    r4.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mul (16|M0)              acc0.0<1>:f   r6.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        mul (16|M16)             r18.0<1>:f    r9.0<8;8,1>:f     9.765625e-04:f               {Compacted,@4}
        add (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   -r11.0<8;8,1>:f  {Compacted,@4}
        add (16|M16)             r18.0<1>:f    r18.0<8;8,1>:f    -r13.0<8;8,1>:f  {Compacted,@2}
        mad (16|M0)              acc0.0<1>:f   r11.0<8;1>:f      acc0.0<8;1>:f     r98.0<1>:f       {Compacted}
        mad (16|M16)             r22.0<1>:f    r13.0<8;1>:f      r18.0<8;1>:f      r100.0<1>:f      {Compacted,@2}
        mul (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   1024.0:f               {Compacted}
        mul (16|M16)             r22.0<1>:f    r22.0<8;8,1>:f    1024.0:f               {Compacted,@2}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r26.0<1>:f    r22.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r123.0<1>:f   acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r125.0<1>:f   r26.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
        else (32|M0)                         L3216                  L3216                
L2712:
(W)     mov (1|M0)               r8.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
        add (16|M0)              r6.0<1>:d     r115.0<8;8,1>:d   r106.0<8;8,1>:d  {Compacted}
        add (16|M16)             r9.0<1>:d     r117.0<8;8,1>:d   r108.0<8;8,1>:d  {Compacted}
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r94.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M0)              r2.0<1>:d     r94.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r95.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M8)              r3.0<1>:d     r95.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r96.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M16)             r4.0<1>:d     r96.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r97.0<8;8,1>:d    r8.10<0;1,0>:uw 
        mach (8|M24)             r5.0<1>:d     r97.0<8;8,1>:d    r8.5<0;1,0>:d    {Compacted}
        add (16|M0)              r11.0<1>:d    r2.0<8;8,1>:d     r102.0<8;8,1>:d  {Compacted,@5}
        sync.nop                             null                             {Compacted,$2.src}
        add (16|M16)             r13.0<1>:d    r4.0<8;8,1>:d     r104.0<8;8,1>:d  {Compacted,@2}
        add (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     r106.0<8;8,1>:d  {Compacted}
        add (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     r108.0<8;8,1>:d  {Compacted}
        shl (16|M0)              r6.0<1>:d     r6.0<8;8,1>:d     2:w               {Compacted}
        shl (16|M16)             r9.0<1>:d     r9.0<8;8,1>:d     2:w               {Compacted}
        shl (16|M0)              r11.0<1>:d    r11.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r13.0<1>:d    r13.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     2:w               {Compacted,@6}
        shl (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     2:w               {Compacted,@6}
        sync.nop                             null                             {Compacted,$3.src}
        send.dc1 (16|M0)         r15      r6      null    0x0            0x04205E00           {@6,$10} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r17      r9      null    0x0            0x04205E00           {@5,$11} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r19      r11     null    0x0            0x04205E00           {@4,$12} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r21      r13     null    0x0            0x04205E00           {@3,$13} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r24      r2      null    0x0            0x04205E00           {@2,$14} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r4      null    0x0            0x04205E00           {@1,$15} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r8.0<0;1,0>:ud                  
        sync.nop                             null                             {Compacted,$10.dst}
        add (16|M0)              acc0.0<1>:f   r15.0<8;8,1>:f    -r119.0<8;8,1>:f {Compacted,$2.dst}
        sync.nop                             null                             {Compacted,$11.dst}
        add (16|M16)             r30.0<1>:f    r17.0<8;8,1>:f    -r121.0<8;8,1>:f {Compacted,$3.dst}
        mad (16|M0)              r33.0<1>:f    r119.0<8;1>:f     acc0.0<8;1>:f     r111.0<1>:f      {Compacted}
        mad (16|M16)             r35.0<1>:f    r121.0<8;1>:f     r30.0<8;1>:f      r113.0<1>:f      {Compacted,@2}
        sync.nop                             null                             {Compacted,$14.dst}
        add (16|M0)              acc0.0<1>:f   r24.0<8;8,1>:f    -r19.0<8;8,1>:f  {Compacted,$12.dst}
        sync.nop                             null                             {Compacted,$15.dst}
        add (16|M16)             r26.0<1>:f    r26.0<8;8,1>:f    -r21.0<8;8,1>:f  {Compacted,$13.dst}
        mad (16|M0)              acc0.0<1>:f   r19.0<8;1>:f      acc0.0<8;1>:f     r111.0<1>:f      {Compacted}
        mad (16|M16)             r40.0<1>:f    r21.0<8;1>:f      r26.0<8;1>:f      r113.0<1>:f      {Compacted,@2}
        add (16|M0)              acc0.0<1>:f   acc0.0<8;8,1>:f   -r33.0<8;8,1>:f  {Compacted,@6}
        add (16|M16)             r40.0<1>:f    r40.0<8;8,1>:f    -r35.0<8;8,1>:f  {Compacted,@2}
        mad (16|M0)              r123.0<1>:f   r33.0<8;1>:f      acc0.0<8;1>:f     r98.0<1>:f       {Compacted}
        mad (16|M16)             r125.0<1>:f   r35.0<8;1>:f      r40.0<8;1>:f      r100.0<1>:f      {Compacted,@2}
L3216:
        endif (32|M0)                        L3232                                
L3232:
(W)     mov (1|M0)               r8.0<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r90.0<8;8,1>:d    r8.14<0;1,0>:uw 
        mach (8|M0)              r2.0<1>:d     r90.0<8;8,1>:d    r8.7<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r91.0<8;8,1>:d    r8.14<0;1,0>:uw 
        mach (8|M8)              r3.0<1>:d     r91.0<8;8,1>:d    r8.7<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r92.0<8;8,1>:d    r8.14<0;1,0>:uw 
        mach (8|M16)             r4.0<1>:d     r92.0<8;8,1>:d    r8.7<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r93.0<8;8,1>:d    r8.14<0;1,0>:uw 
        mach (8|M24)             r5.0<1>:d     r93.0<8;8,1>:d    r8.7<0;1,0>:d    {Compacted}
        add (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     r86.0<8;8,1>:d   {Compacted,@5}
        add (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     r88.0<8;8,1>:d   {Compacted,@2}
        shl (16|M0)              r2.0<1>:d     r2.0<8;8,1>:d     2:w               {Compacted,@2}
        shl (16|M16)             r4.0<1>:d     r4.0<8;8,1>:d     2:w               {Compacted,@2}
        send.dc1 (16|M0)         null     r2      r123    0x80            0x04025E01           {@2,$0} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r4      r125    0x80            0x04025E01           {@1,$1} // wr:2+2, rd:0; untyped surface write with x
(W)     mov (1|M0)               f0.0<1>:ud    r8.0<0;1,0>:ud                  
L3464:
        endif (32|M0)                        L3480                                
L3480:
(W)     mov (8|M0)               r127.0<1>:f   r85.0<8;8,1>:f                   {Compacted}
        sync.nop                             null                             {Compacted,$0.src}
(W)     send.dc0 (8|M0)          r2       r85     null    0x0            0x0219E000           {$2} // wr:1h+0, rd:1; synchronized global fence flushing
(W)     mov (8|M0)               null<1>:ud    r2.0<8;8,1>:ud                   {$2.dst}
(W)     mov (16|M0)              acc0.0<1>:f   0.0:f                              
(W)     send.ts (1|M0)           null     r127    null    0x0            0x02000010           {EOT,@1} // wr:1+0, rd:0; end of thread
L3560:
        nop                    
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
