L0:
(W)     mov (8|M0)               r32.0<1>:ud   r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {@1}
(W)     mul (1|M0)               acc0.0<1>:d   r10.3<0;1,0>:d    r32.2<0;1,0>:uw  {@1}
(W)     mach (1|M0)              r5.0<1>:d     r10.3<0;1,0>:d    r32.1<0;1,0>:d  
(W)     mul (1|M0)               acc0.0<1>:d   r10.4<0;1,0>:d    r32.12<0;1,0>:uw
        mov (16|M0)              r34.0<1>:d    r1.0<16;16,1>:uw                
        mov (16|M16)             r36.0<1>:d    r2.0<16;16,1>:uw                
        mov (16|M0)              r42.0<1>:d    r3.0<16;16,1>:uw                
        mov (16|M16)             r44.0<1>:d    r4.0<16;16,1>:uw                
(W)     mach (1|M0)              r6.0<1>:d     r10.4<0;1,0>:d    r32.6<0;1,0>:d  
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r9.4<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r9.4<0;1,0>:d     0:w              
        add (16|M0)              r12.0<1>:d    r5.0<0;1,0>:d     r34.0<8;8,1>:d   {Compacted,@7}
        add (16|M16)             r14.0<1>:d    r5.0<0;1,0>:d     r36.0<8;8,1>:d   {Compacted,@7}
        add (16|M0)              r16.0<1>:d    r6.0<0;1,0>:d     r42.0<8;8,1>:d   {Compacted,@5}
        add (16|M16)             r18.0<1>:d    r6.0<0;1,0>:d     r44.0<8;8,1>:d   {Compacted,@7}
        add (16|M0)              r38.0<1>:d    r12.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@4}
        add (16|M16)             r40.0<1>:d    r14.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@4}
        add (16|M0)              r46.0<1>:d    r16.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@4}
        add (16|M16)             r48.0<1>:d    r18.0<8;8,1>:d    r7.1<0;1,0>:d    {Compacted,@4}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f0.0)  if (32|M0)                           L312                  L6056                
L280:
        mov (16|M0)              r50.0<1>:f    0.0:f                               {Compacted}
        mov (16|M16)             r52.0<1>:f    0.0:f                               {Compacted}
        else (32|M0)                         L6056                  L6056                
L312:
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r10.0<0;1,0>:d    0:w              
(W)     add (1|M0)               r5.0<1>:d     r10.0<0;1,0>:d    -1:w               {Compacted}
(W)     mul (8|M0)               acc0.0<1>:d   r46.0<8;8,1>:d    r9.8<0;1,0>:uw   {Compacted,@7}
(W)     mov (1|M0)               r33.2<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r5.0<0;1,0>:ud    0x3:uw              {@3}
        mach (8|M0)              r54.0<1>:d    r46.0<8;8,1>:d    r9.4<0;1,0>:d    {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r47.0<8;8,1>:d    r9.8<0;1,0>:uw  
(W)     mov (1|M0)               r33.3<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
        mach (8|M8)              r55.0<1>:d    r47.0<8;8,1>:d    r9.4<0;1,0>:d    {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r48.0<8;8,1>:d    r9.8<0;1,0>:uw  
(W)     mov (1|M0)               f1.0<1>:ud    r33.3<0;1,0>:ud                  {@3}
        mach (8|M16)             r56.0<1>:d    r48.0<8;8,1>:d    r9.4<0;1,0>:d    {Compacted}
(W)     cmp (16|M16)  (lt)f1.0   null<1>:d     r5.0<0;1,0>:ud    0x3:uw             
(W)     mul (8|M24)              acc0.0<1>:d   r49.0<8;8,1>:d    r9.8<0;1,0>:uw  
        mach (8|M24)             r57.0<1>:d    r49.0<8;8,1>:d    r9.4<0;1,0>:d    {Compacted}
(W)     mul (8|M0)               acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw  {Compacted}
(W)     and (1|M0)               r9.0<1>:d     r10.0<0;1,0>:d    3:w               {Compacted}
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W)     mul (8|M8)               acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W)     mul (8|M16)              acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
(W)     mov (1|M0)               r33.3<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r9.0<0;1,0>:d     0:w               {@7}
(W)     mul (8|M24)              acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
        add (16|M0)              r58.0<1>:d    r12.0<8;8,1>:d    r34.0<8;8,1>:d   {Compacted,@7}
        add (16|M16)             r60.0<1>:d    r14.0<8;8,1>:d    r36.0<8;8,1>:d   {Compacted,@2}
(W)     mov (1|M0)               r10.7<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    r33.2<0;1,0>:ud                 
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r10.0<0;1,0>:d    0:w              
        shl (16|M0)              r12.0<1>:d    r34.0<8;8,1>:d    2:w               {Compacted}
        shl (16|M0)              r58.0<1>:d    r58.0<8;8,1>:d    2:w               {Compacted,@6}
        shl (16|M16)             r14.0<1>:d    r36.0<8;8,1>:d    2:w               {Compacted}
        shl (16|M16)             r60.0<1>:d    r60.0<8;8,1>:d    2:w               {Compacted,@7}
(W)     mov (1|M0)               f1.0<1>:ud    r10.7<0;1,0>:ud                  {@7}
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r9.0<0;1,0>:d     0:w              
        mov (16|M0)              r50.0<1>:f    0.0:f                               {Compacted}
        mov (16|M16)             r52.0<1>:f    0.0:f                               {Compacted}
(W)     mov (1|M0)               r33.2<1>:ud   f0.0<0;1,0>:ud                   {Compacted}
        add (16|M0)              r66.0<1>:d    r12.0<8;8,1>:d    r8.2<0;1,0>:d    {Compacted,@7}
        add (16|M0)              r62.0<1>:d    r58.0<8;8,1>:d    r8.2<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r68.0<1>:d    r14.0<8;8,1>:d    r8.2<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r64.0<1>:d    r60.0<8;8,1>:d    r8.2<0;1,0>:d    {Compacted,@7}
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r9.5<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r9.5<0;1,0>:d     0:w              
(W)     and (1|M0)               r9.1<1>:d     r10.0<0;1,0>:d    -4:w               {Compacted}
(W)     mov (1|M0)               r9.6<1>:f     0.0:f                               {Compacted}
(W)     mov (1|M0)               r10.7<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
L832:
        add (16|M0)              r12.0<1>:d    r9.6<0;1,0>:d     r34.0<8;8,1>:d   {Compacted,@2}
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
        cmp (16|M0)   (lt)f1.0   null<1>:d     r12.0<8;8,1>:ud   r9.4<0;1,0>:ud   {@2}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
        add (16|M16)             r14.0<1>:d    r9.6<0;1,0>:d     r36.0<8;8,1>:d   {Compacted}
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
        cmp (16|M16)  (lt)f1.0   null<1>:d     r14.0<8;8,1>:ud   r9.4<0;1,0>:ud   {@2}
(W&f0.0.any32h) mov (1|M0)       r5.4<1>:hf    0x1:hf                             
        add (16|M0)              r70.0<1>:d    r9.6<0;1,0>:d     r42.0<8;8,1>:d   {Compacted}
(W&f0.0.any32h) mov (1|M0)       r5.1<1>:ud    f1.0<0;1,0>:ud                  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@5}
        add (16|M16)             r72.0<1>:d    r9.6<0;1,0>:d     r44.0<8;8,1>:d   {Compacted}
(f1.0)  cmp (16|M0)   (lt)f1.0   null<1>:d     r46.0<8;8,1>:ud   r9.2<0;1,0>:ud  
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@1}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r48.0<8;8,1>:ud   r9.2<0;1,0>:ud  
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(f1.0)  sel (16|M16)             acc0.0<1>:uw  r5.4<0;1,0>:uw    0x0:uw              {@7}
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.1<0;1,0>:ud                   {@7}
(f1.0)  sel (16|M16)             r11.0<1>:uw   r5.4<0;1,0>:uw    0x0:uw             
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@4}
        and (16|M16)  (ne)f1.0   null<1>:uw    acc0.0<16;16,1>:uw  r11.0<16;16,1>:uw {@2}
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) if (32|M0)                           L1256                  L1352                
L1224:
        mov (16|M0)              r74.0<1>:f    0.0:f                               {Compacted,$2.src}
        mov (16|M16)             r76.0<1>:f    0.0:f                               {Compacted,$0.src}
        else (32|M0)                         L1352                  L1352                
L1256:
        add (16|M0)              r12.0<1>:d    r54.0<8;8,1>:d    r9.6<0;1,0>:d    {Compacted}
        add (16|M16)             r14.0<1>:d    r56.0<8;8,1>:d    r9.6<0;1,0>:d    {Compacted}
        add (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    r34.0<8;8,1>:d   {Compacted,@2}
        add (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    r36.0<8;8,1>:d   {Compacted,@2}
        shl (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    2:w               {Compacted,@2}
        sync.nop                             null                             {Compacted,$2.src}
        send.dc1 (16|M0)         r74      r12     null    0x0            0x04205E00           {@2,$5} // wr:2+0, rd:2; untyped surface read with x
        sync.nop                             null                             {Compacted,$0.src}
        send.dc1 (16|M16)        r76      r14     null    0x0            0x04205E00           {@1,$6} // wr:2+0, rd:2; untyped surface read with x
L1352:
        endif (32|M0)                        L1368                                
L1368:
        cmp (16|M0)   (lt)f1.0   null<1>:d     r70.0<8;8,1>:ud   r9.4<0;1,0>:ud  
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
        cmp (16|M16)  (lt)f1.0   null<1>:d     r72.0<8;8,1>:ud   r9.4<0;1,0>:ud  
(W&f0.0.any32h) mov (1|M0)       r5.4<1>:hf    0x1:hf                             
        sync.nop                             null                             {Compacted,$5.dst}
        send.dc1 (16|M0)         null     r58     r74     0x80            0x04025EFE           {$2} // wr:2+2, rd:0; untyped surface write with x
(W&f0.0.any32h) mov (1|M0)       r5.1<1>:ud    f1.0<0;1,0>:ud                  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@4}
        sync.nop                             null                             {Compacted,$6.dst}
        send.dc1 (16|M16)        null     r60     r76     0x80            0x04025EFE           {$0} // wr:2+2, rd:0; untyped surface write with x
(f1.0)  cmp (16|M0)   (lt)f1.0   null<1>:d     r38.0<8;8,1>:ud   r9.3<0;1,0>:ud  
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@1}
        cmp (16|M16)  (lt)f1.0   null<1>:d     r40.0<8;8,1>:ud   r9.3<0;1,0>:ud  
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(f1.0)  sel (16|M16)             acc0.0<1>:uw  r5.4<0;1,0>:uw    0x0:uw              {@7}
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.1<0;1,0>:ud                   {@7}
(f1.0)  sel (16|M16)             r11.0<1>:uw   r5.4<0;1,0>:uw    0x0:uw             
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r5.0<0;1,0>:ud                   {@4}
        and (16|M16)  (ne)f1.0   null<1>:uw    acc0.0<16;16,1>:uw  r11.0<16;16,1>:uw {@2}
(W&f0.0.any32h) mov (1|M0)       r5.0<1>:ud    f1.0<0;1,0>:ud                  
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) if (32|M0)                           L1808                  L2056                
L1776:
        mov (16|M0)              r78.0<1>:f    0.0:f                               {Compacted,$4.src}
        mov (16|M16)             r80.0<1>:f    0.0:f                               {Compacted,$3.src}
        else (32|M0)                         L2056                  L2056                
L1808:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r70.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M0)              r12.0<1>:d    r70.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r71.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M8)              r13.0<1>:d    r71.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r72.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M16)             r14.0<1>:d    r72.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r73.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M24)             r15.0<1>:d    r73.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
        add (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    r38.0<8;8,1>:d   {Compacted,@5}
        add (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    r40.0<8;8,1>:d   {Compacted,@2}
        shl (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    2:w               {Compacted,@2}
        sync.nop                             null                             {Compacted,$4.src}
        send.dc1 (16|M0)         r78      r12     null    0x0            0x04205E01           {@2,$7} // wr:2+0, rd:2; untyped surface read with x
        sync.nop                             null                             {Compacted,$3.src}
        send.dc1 (16|M16)        r80      r14     null    0x0            0x04205E01           {@1,$8} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
L2056:
        endif (32|M0)                        L2072                                
L2072:
        sync.nop                             null                             {Compacted,$7.dst}
        send.dc1 (16|M0)         null     r62     r78     0x80            0x04025EFE           {$4} // wr:2+2, rd:0; untyped surface write with x
        sync.nop                             null                             {Compacted,$8.dst}
        send.dc1 (16|M16)        null     r64     r80     0x80            0x04025EFE           {$3} // wr:2+2, rd:0; untyped surface write with x
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(W&f0.0.any32h) send.dc0 (8|M0)   r5      r32     null    0x0            0x0219E0FE           {$9} // wr:1h+0, rd:1; synchronized SLM fence
(W&f0.0.any32h) and (8|M0)       r6.0<1>:ud    r32.2<0;1,0>:ud   0x7F000000:ud              {$1.src}
(W)     mov (8|M0)               null<1>:ud    r5.0<8;8,1>:ud                   {$9.dst}
(W&f0.0.any32h) send.gtwy (1|M0)   null   r6      null    0x0            0x02000004           {@2,$10} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             null                            
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r33.2<0;1,0>:ud                 
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@7}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@5}
(~f1.0) if (32|M0)                           L5744                  L5744                
L2320:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r33.3<0;1,0>:ud                 
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@4}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(~f1.0) if (32|M0)                           L4672                  L4760                
L2424:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (2|M0)       r33.0<1>:f    0.0:f                              
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@4}
L2496:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw 
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (1|M0)              acc0.0<1>:d   r33.0<0;1,0>:d    r10.0<0;1,0>:uw 
(W&f0.0) mach (1|M0)             r5.0<1>:d     r33.0<0;1,0>:d    r10.0<0;1,0>:d  
        add (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    r33.0<0;1,0>:d   {Compacted,@7}
        add (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    r33.0<0;1,0>:d   {Compacted,@4}
        sync.nop                             null                             {Compacted,$10.src}
(W&f0.0) shl (1|M0)              r6.0<1>:d     r5.0<0;1,0>:d     2:w               {@3}
        shl (16|M0)              r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@3}
        shl (16|M16)             r18.0<1>:d    r18.0<8;8,1>:d    2:w               {Compacted,@3}
        add (16|M0)              r20.0<1>:d    r66.0<8;8,1>:d    r6.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r22.0<1>:d    r68.0<8;8,1>:d    r6.0<0;1,0>:d    {Compacted}
        send.dc1 (16|M0)         r24      r16     null    0x0            0x04205EFE           {@4,$11} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r18     null    0x0            0x04205EFE           {@3,$12} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r28      r20     null    0x0            0x04205EFE           {@2,$13} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r30      r22     null    0x0            0x04205EFE           {@1,$14} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
        sync.nop                             null                             {Compacted,$13.dst}
        mad (16|M0)              r82.0<1>:f    r50.0<8;1>:f      r28.0<8;1>:f      r24.0<1>:f       {Compacted,$11.dst}
        sync.nop                             null                             {Compacted,$14.dst}
        mad (16|M16)             r84.0<1>:f    r52.0<8;1>:f      r30.0<8;1>:f      r26.0<1>:f       {Compacted,$12.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L2952                  L2952                
L2896:
        mul (16|M0)              acc0.0<1>:f   r82.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r14.0<1>:f    r84.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r82.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r84.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L2952:
        endif (32|M0)                        L2968                                
L2968:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw 
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
(W&f0.0) or (1|M0)               r5.0<1>:d     r33.0<0;1,0>:d    1:w              
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (1|M0)              acc0.0<1>:d   r5.0<0;1,0>:d     r10.0<0;1,0>:uw  {@6}
(W&f0.0) mach (1|M0)             r6.0<1>:d     r5.0<0;1,0>:d     r10.0<0;1,0>:d  
        add (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@4}
(W&f0.0) shl (1|M0)              r8.0<1>:d     r6.0<0;1,0>:d     2:w               {@3}
        shl (16|M0)              r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@3}
        shl (16|M16)             r18.0<1>:d    r18.0<8;8,1>:d    2:w               {Compacted,@3}
        add (16|M0)              r20.0<1>:d    r66.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r22.0<1>:d    r68.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted}
        send.dc1 (16|M0)         r24      r16     null    0x0            0x04205EFE           {@4,$15} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r18     null    0x0            0x04205EFE           {@3,$5} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r28      r20     null    0x0            0x04205EFE           {@2,$6} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r30      r22     null    0x0            0x04205EFE           {@1,$7} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
        sync.nop                             null                             {Compacted,$6.dst}
        mad (16|M0)              r86.0<1>:f    r82.0<8;1>:f      r28.0<8;1>:f      r24.0<1>:f       {Compacted,$15.dst}
        sync.nop                             null                             {Compacted,$7.dst}
        mad (16|M16)             r88.0<1>:f    r84.0<8;1>:f      r30.0<8;1>:f      r26.0<1>:f       {Compacted,$5.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L3432                  L3432                
L3376:
        mul (16|M0)              acc0.0<1>:f   r86.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r14.0<1>:f    r88.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r86.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r88.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L3432:
        endif (32|M0)                        L3448                                
L3448:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw 
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
(W&f0.0) or (1|M0)               r5.0<1>:d     r33.0<0;1,0>:d    2:w              
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (1|M0)              acc0.0<1>:d   r5.0<0;1,0>:d     r10.0<0;1,0>:uw  {@6}
(W&f0.0) mach (1|M0)             r6.0<1>:d     r5.0<0;1,0>:d     r10.0<0;1,0>:d  
        add (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@4}
(W&f0.0) shl (1|M0)              r8.0<1>:d     r6.0<0;1,0>:d     2:w               {@3}
        shl (16|M0)              r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@3}
        shl (16|M16)             r18.0<1>:d    r18.0<8;8,1>:d    2:w               {Compacted,@3}
        add (16|M0)              r20.0<1>:d    r66.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r22.0<1>:d    r68.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted}
        send.dc1 (16|M0)         r24      r16     null    0x0            0x04205EFE           {@4,$8} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r18     null    0x0            0x04205EFE           {@3,$9} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r28      r20     null    0x0            0x04205EFE           {@2,$11} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r30      r22     null    0x0            0x04205EFE           {@1,$12} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
        sync.nop                             null                             {Compacted,$11.dst}
        mad (16|M0)              r90.0<1>:f    r86.0<8;1>:f      r28.0<8;1>:f      r24.0<1>:f       {Compacted,$8.dst}
        sync.nop                             null                             {Compacted,$12.dst}
        mad (16|M16)             r92.0<1>:f    r88.0<8;1>:f      r30.0<8;1>:f      r26.0<1>:f       {Compacted,$9.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L3912                  L3912                
L3856:
        mul (16|M0)              acc0.0<1>:f   r90.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r14.0<1>:f    r92.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r90.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r92.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L3912:
        endif (32|M0)                        L3928                                
L3928:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw 
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
(W&f0.0) or (1|M0)               r5.0<1>:d     r33.0<0;1,0>:d    3:w              
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (1|M0)              acc0.0<1>:d   r5.0<0;1,0>:d     r10.0<0;1,0>:uw  {@6}
(W&f0.0) mach (1|M0)             r6.0<1>:d     r5.0<0;1,0>:d     r10.0<0;1,0>:d  
        add (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    r5.0<0;1,0>:d    {Compacted,@4}
(W&f0.0) shl (1|M0)              r8.0<1>:d     r6.0<0;1,0>:d     2:w               {@3}
        shl (16|M0)              r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@3}
        shl (16|M16)             r18.0<1>:d    r18.0<8;8,1>:d    2:w               {Compacted,@3}
        add (16|M0)              r20.0<1>:d    r66.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r22.0<1>:d    r68.0<8;8,1>:d    r8.0<0;1,0>:d    {Compacted}
        send.dc1 (16|M0)         r24      r16     null    0x0            0x04205EFE           {@4,$13} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r18     null    0x0            0x04205EFE           {@3,$14} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r28      r20     null    0x0            0x04205EFE           {@2,$15} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r30      r22     null    0x0            0x04205EFE           {@1,$5} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
        sync.nop                             null                             {Compacted,$15.dst}
        mad (16|M0)              r50.0<1>:f    r90.0<8;1>:f      r28.0<8;1>:f      r24.0<1>:f       {Compacted,$13.dst}
        sync.nop                             null                             {Compacted,$5.dst}
        mad (16|M16)             r52.0<1>:f    r92.0<8;1>:f      r30.0<8;1>:f      r26.0<1>:f       {Compacted,$14.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L4392                  L4392                
L4336:
        mul (16|M0)              acc0.0<1>:f   r50.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r14.0<1>:f    r52.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r50.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r52.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L4392:
        endif (32|M0)                        L4408                                
L4408:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W&f0.0) add (1|M0)              r33.1<1>:d    r33.1<0;1,0>:d    4:w              
(W&f0.0) add (1|M0)              r33.0<1>:d    r33.0<0;1,0>:d    4:w              
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r33.1<0;1,0>:d    r9.1<0;1,0>:d    {@2}
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@4}
(W&f0.0) mov (1|M0)              r9.7<1>:f     r33.0<0;1,0>:f                   {@3}
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r33.1<0;1,0>:d    r9.1<0;1,0>:d   
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@2}
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) while (32|M0)                        L2496                                
L4640:
        mov (16|M0)              r94.0<1>:f    r50.0<8;8,1>:f                   {Compacted}
        mov (16|M16)             r96.0<1>:f    r52.0<8;8,1>:f                   {Compacted}
        else (32|M0)                         L4760                  L4760                
L4672:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
        mov (16|M0)              r94.0<1>:f    r50.0<8;8,1>:f                   {Compacted}
        mov (16|M16)             r96.0<1>:f    r52.0<8;8,1>:f                   {Compacted}
(W&f0.0.any32h) mov (1|M0)       r9.7<1>:f     0.0:f                              
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@6}
L4760:
        endif (32|M0)                        L4776                                
L4776:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f1.0<1>:ud    r10.7<0;1,0>:ud                 
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@4}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) if (32|M0)                           L5728                  L5728                
L4880:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
        mov (16|M0)              r50.0<1>:f    r94.0<8;8,1>:f                   {Compacted}
        mov (16|M16)             r52.0<1>:f    r96.0<8;8,1>:f                   {Compacted}
(W&f0.0.any32h) mov (1|M0)       r10.6<1>:f    0.0:f                              
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@6}
L4968:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r10.0<0;1,0>:d    r42.0<2;1,0>:uw 
        mach (8|M0)              r12.0<1>:d    r10.0<0;1,0>:d    r42.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r10.0<0;1,0>:d    r43.0<2;1,0>:uw 
        mach (8|M8)              r13.0<1>:d    r10.0<0;1,0>:d    r43.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r10.0<0;1,0>:d    r44.0<2;1,0>:uw 
        mach (8|M16)             r14.0<1>:d    r10.0<0;1,0>:d    r44.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r10.0<0;1,0>:d    r45.0<2;1,0>:uw 
        mach (8|M24)             r15.0<1>:d    r10.0<0;1,0>:d    r45.0<8;8,1>:d   {Compacted}
(W&f0.0) mul (1|M0)              acc0.0<1>:d   r9.7<0;1,0>:d     r10.0<0;1,0>:uw 
(W&f0.0) mach (1|M0)             r5.0<1>:d     r9.7<0;1,0>:d     r10.0<0;1,0>:d  
        add (16|M0)              r16.0<1>:d    r12.0<8;8,1>:d    r9.7<0;1,0>:d    {Compacted,@7}
        add (16|M16)             r18.0<1>:d    r14.0<8;8,1>:d    r9.7<0;1,0>:d    {Compacted,@4}
        sync.nop                             null                             {Compacted,$10.src}
(W&f0.0) shl (1|M0)              r6.0<1>:d     r5.0<0;1,0>:d     2:w               {@3}
        shl (16|M0)              r16.0<1>:d    r16.0<8;8,1>:d    2:w               {Compacted,@3}
        shl (16|M16)             r18.0<1>:d    r18.0<8;8,1>:d    2:w               {Compacted,@3}
        add (16|M0)              r20.0<1>:d    r66.0<8;8,1>:d    r6.0<0;1,0>:d    {Compacted,@3}
        add (16|M16)             r22.0<1>:d    r68.0<8;8,1>:d    r6.0<0;1,0>:d    {Compacted}
        send.dc1 (16|M0)         r24      r16     null    0x0            0x04205EFE           {@4,$6} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r26      r18     null    0x0            0x04205EFE           {@3,$7} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r28      r20     null    0x0            0x04205EFE           {@2,$8} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r30      r22     null    0x0            0x04205EFE           {@1,$9} // wr:2+0, rd:2; untyped surface read with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
        sync.nop                             null                             {Compacted,$8.dst}
        mad (16|M0)              r50.0<1>:f    r50.0<8;1>:f      r28.0<8;1>:f      r24.0<1>:f       {Compacted,$6.dst}
        sync.nop                             null                             {Compacted,$9.dst}
        mad (16|M16)             r52.0<1>:f    r52.0<8;1>:f      r30.0<8;1>:f      r26.0<1>:f       {Compacted,$7.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L5424                  L5424                
L5368:
        mul (16|M0)              acc0.0<1>:f   r50.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r14.0<1>:f    r52.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r50.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r52.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L5424:
        endif (32|M0)                        L5440                                
L5440:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W&f0.0) add (1|M0)              r10.6<1>:d    r10.6<0;1,0>:d    1:w              
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r10.6<0;1,0>:d    r9.0<0;1,0>:d    {@1}
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@3}
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r10.6<0;1,0>:d    r9.0<0;1,0>:d   
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@2}
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@7}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f1.0)  break (32|M0)                        L5712                  L5712                
L5640:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) add (1|M0)       r9.7<1>:d     r9.7<0;1,0>:d     1:w              
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                 {@4}
L5712:
        while (32|M0)                        L4968                                
L5728:
        endif (32|M0)                        L5744                                
L5744:
        endif (32|M0)                        L5760                                
L5760:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(W&f0.0) send.dc0 (8|M0)         r5       r32     null    0x0            0x0219E0FE           {$11} // wr:1h+0, rd:1; synchronized SLM fence
(W&f0.0) and (8|M0)              r6.0<1>:ud    r32.2<0;1,0>:ud   0x7F000000:ud              {$10.src}
(W)     mov (8|M0)               null<1>:ud    r5.0<8;8,1>:ud                   {$11.dst}
(W&f0.0) send.gtwy (1|M0)        null     r6      null    0x0            0x02000004           {@2,$1} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             null                            
(W&f0.0) add (1|M0)              r9.6<1>:d     r9.6<0;1,0>:d     r10.0<0;1,0>:d  
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r9.6<0;1,0>:ud    r9.4<0;1,0>:ud   {@2}
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@2}
(W)     mov (1|M0)               r98.0<1>:ud   f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M16)  (lt)f1.0   null<1>:d     r9.6<0;1,0>:ud    r9.4<0;1,0>:ud  
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r98.0<0;1,0>:ud                  {@2}
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(f1.0)  while (32|M0)                        L832                                
L6056:
        endif (32|M0)                        L6072                                
L6072:
        cmp (16|M16)  (lt)f1.0   null<1>:d     r40.0<8;8,1>:ud   r9.3<0;1,0>:ud  
        cmp (16|M16)  (lt)f0.0   null<1>:d     r48.0<8;8,1>:ud   r9.2<0;1,0>:ud  
        cmp (16|M0)   (lt)f0.0   null<1>:d     r38.0<8;8,1>:ud   r9.3<0;1,0>:ud  
(W)     mov (1|M0)               r5.0<1>:hf    0x1:hf                             
(f0.0)  sel (16|M16)             acc0.0<1>:uw  r5.0<0;1,0>:uw    0x0:uw              {@1}
(f1.0)  sel (16|M16)             r11.0<1>:uw   r5.0<0;1,0>:uw    0x0:uw             
(f0.0)  cmp (16|M0)   (lt)f0.0   null<1>:d     r46.0<8;8,1>:ud   r9.2<0;1,0>:ud  
        and (16|M16)  (ne)f0.0   null<1>:uw    acc0.0<16;16,1>:uw  r11.0<16;16,1>:uw {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(f0.0)  if (32|M0)                           L6600                  L6600                
L6232:
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r9.5<0;1,0>:d     0:w              
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r9.5<0;1,0>:d     0:w              
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@4}
(~f1.0) if (32|M0)                           L6352                  L6352                
L6296:
        mul (16|M0)              acc0.0<1>:f   r50.0<8;8,1>:f    1024.0:f               {Compacted}
        mul (16|M16)             r14.0<1>:f    r52.0<8;8,1>:f    1024.0:f               {Compacted}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r18.0<1>:f    r14.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r50.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r52.0<1>:f    r18.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L6352:
        endif (32|M0)                        L6368                                
L6368:
(W)     mov (1|M0)               r100.0<1>:ud  f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W&f0.0) mul (8|M0)              acc0.0<1>:d   r46.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M0)              r12.0<1>:d    r46.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M8)              acc0.0<1>:d   r47.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M8)              r13.0<1>:d    r47.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M16)             acc0.0<1>:d   r48.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M16)             r14.0<1>:d    r48.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
(W&f0.0) mul (8|M24)             acc0.0<1>:d   r49.0<8;8,1>:d    r9.6<0;1,0>:uw  
        mach (8|M24)             r15.0<1>:d    r49.0<8;8,1>:d    r9.3<0;1,0>:d    {Compacted}
        add (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    r38.0<8;8,1>:d   {Compacted,@5}
        add (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    r40.0<8;8,1>:d   {Compacted,@2}
        shl (16|M0)              r12.0<1>:d    r12.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r14.0<1>:d    r14.0<8;8,1>:d    2:w               {Compacted,@2}
        send.dc1 (16|M0)         null     r12     r50     0x80            0x04025E02           {@2,$12} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r14     r52     0x80            0x04025E02           {@1,$13} // wr:2+2, rd:0; untyped surface write with x
(W)     mov (1|M0)               f0.0<1>:ud    r100.0<0;1,0>:ud                
L6600:
        endif (32|M0)                        L6616                                
L6616:
(W)     mov (8|M0)               r127.0<1>:f   r32.0<8;8,1>:f                   {Compacted}
(W)     send.dc0 (8|M0)          r5       r32     null    0x0            0x0219E000           {$14} // wr:1h+0, rd:1; synchronized global fence flushing
(W)     mov (8|M0)               null<1>:ud    r5.0<8;8,1>:ud                   {$14.dst}
        sync.nop                             null                             {Compacted,$1.src}
(W)     send.dc0 (8|M0)          r6       r32     null    0x0            0x0219E0FE           {$15} // wr:1h+0, rd:1; synchronized SLM fence
(W)     mov (8|M0)               null<1>:ud    r6.0<8;8,1>:ud                   {$15.dst}
(W)     mov (16|M0)              acc0.0<1>:f   0.0:f                              
(W)     send.ts (1|M0)           null     r127    null    0x0            0x02000010           {EOT,@1} // wr:1+0, rd:0; end of thread
L6728:
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
