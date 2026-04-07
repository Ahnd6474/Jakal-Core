L0:
(W)     mov (8|M0)               r3.0<1>:ud    r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {@1}
(W)     mul (1|M0)               acc0.0<1>:d   r10.0<0;1,0>:d    r3.2<0;1,0>:uw   {Compacted,@1}
        mov (16|M0)              r5.0<1>:d     r1.0<16;16,1>:uw                
        mov (16|M16)             r11.0<1>:d    r2.0<16;16,1>:uw                
(W)     mach (1|M0)              r4.0<1>:d     r10.0<0;1,0>:d    r3.1<0;1,0>:d    {Compacted}
        add (16|M0)              r13.0<1>:d    r4.0<0;1,0>:d     r5.0<8;8,1>:d    {Compacted,@1}
        add (16|M16)             r15.0<1>:d    r4.0<0;1,0>:d     r11.0<8;8,1>:d   {Compacted,@3}
        add (16|M0)              r17.0<1>:d    r13.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@2}
        add (16|M16)             r19.0<1>:d    r15.0<8;8,1>:d    r7.0<0;1,0>:d    {Compacted,@2}
        cmp (16|M0)   (lt)f0.0   null<1>:d     r17.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
        cmp (16|M16)  (lt)f0.0   null<1>:d     r19.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f0.0) if (32|M0)                           L208                  L504                
L176:
        mov (16|M0)              r21.0<1>:f    0.0:f                               {Compacted}
        mov (16|M16)             r23.0<1>:f    0.0:f                               {Compacted}
        else (32|M0)                         L504                  L504                
L208:
(W)     cmp (16|M0)   (eq)f1.0   null<1>:d     r8.7<0;1,0>:d     0:w              
(W)     cmp (16|M16)  (eq)f1.0   null<1>:d     r8.7<0;1,0>:d     0:w              
        mov (16|M0)              r21.0<1>:f    0.0:f                               {Compacted}
        mov (16|M16)             r23.0<1>:f    0.0:f                               {Compacted}
L256:
        shl (16|M0)              r25.0<1>:d    r17.0<8;8,1>:d    2:w               {Compacted}
        shl (16|M16)             r27.0<1>:d    r19.0<8;8,1>:d    2:w               {Compacted}
        send.dc1 (16|M0)         r29      r25     null    0x0            0x04205E00           {@2,$0} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r31      r27     null    0x0            0x04205E00           {@1,$1} // wr:2+0, rd:2; untyped surface read with x
        add (16|M0)              r21.0<1>:f    r21.0<8;8,1>:f    r29.0<8;8,1>:f   {Compacted,@4,$0.dst}
        add (16|M16)             r23.0<1>:f    r23.0<8;8,1>:f    r31.0<8;8,1>:f   {Compacted,@4,$1.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(~f1.0) if (32|M0)                           L408                  L408                
L352:
        mul (16|M0)              acc0.0<1>:f   r21.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r35.0<1>:f    r23.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r39.0<1>:f    r35.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r21.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r23.0<1>:f    r39.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L408:
        endif (32|M0)                        L424                                
L424:
        add (16|M0)              r17.0<1>:d    r17.0<8;8,1>:d    r9.0<0;1,0>:d    {Compacted}
        add (16|M16)             r19.0<1>:d    r19.0<8;8,1>:d    r9.0<0;1,0>:d    {Compacted}
        cmp (16|M0)   (lt)f0.0   null<1>:d     r17.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
        cmp (16|M16)  (lt)f0.0   null<1>:d     r19.0<8;8,1>:ud   r8.6<0;1,0>:ud   {@2}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f0.0)  while (32|M0)                        L256                                
L504:
        endif (32|M0)                        L520                                
L520:
        shl (16|M0)              r41.0<1>:d    r5.0<8;8,1>:d     2:w               {Compacted}
        shl (16|M16)             r43.0<1>:d    r11.0<8;8,1>:d    2:w               {Compacted}
        send.dc1 (16|M0)         null     r41     r21     0x80            0x04025EFE           {@2,$2} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r43     r23     0x80            0x04025EFE           {@1,$3} // wr:2+2, rd:0; untyped surface write with x
(W)     send.dc0 (8|M0)          r45      r3      null    0x0            0x0219E0FE           {$4} // wr:1h+0, rd:1; synchronized SLM fence
(W)     and (8|M0)               r46.0<1>:ud   r3.2<0;1,0>:ud    0x7F000000:ud             
(W)     mov (8|M0)               null<1>:ud    r45.0<8;8,1>:ud                  {$4.dst}
(W)     send.gtwy (1|M0)         null     r46     null    0x0            0x02000004           {@2,$5} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             null                            
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r9.3<0;1,0>:ud    0x2:uw             
(W)     cmp (16|M16)  (lt)f1.0   null<1>:d     r9.3<0;1,0>:ud    0x2:uw             
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(~f1.0) if (32|M0)                           L1552                  L1552                
L712:
(W)     cmp (16|M0)   (eq)f0.0   null<1>:d     r8.7<0;1,0>:d     0:w               {Compacted}
(W)     cmp (16|M16)  (eq)f0.0   null<1>:d     r8.7<0;1,0>:d     0:w              
(W)     mov (1|M0)               r4.1<1>:f     r9.3<0;1,0>:f                   
L752:
(W)     mov (1|M0)               r4.4<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) shr (1|M0)       r4.2<1>:ud    r4.1<0;1,0>:ud    1:w               {@4}
(W)     mov (1|M0)               f0.0<1>:ud    r4.4<0;1,0>:ud                   {@4}
        cmp (16|M0)   (gt)f1.0   null<1>:d     r4.2<0;1,0>:ud    r5.0<8;8,1>:ud   {@2}
        cmp (16|M16)  (gt)f1.0   null<1>:d     r4.2<0;1,0>:ud    r11.0<8;8,1>:ud 
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f1.0)  if (32|M0)                           L1168                  L1168                
L888:
        add (16|M0)              r51.0<1>:d    r4.2<0;1,0>:d     r5.0<8;8,1>:d    {Compacted}
        add (16|M16)             r53.0<1>:d    r4.2<0;1,0>:d     r11.0<8;8,1>:d   {Compacted}
        shl (16|M0)              r51.0<1>:d    r51.0<8;8,1>:d    2:w               {Compacted,@2}
        shl (16|M16)             r53.0<1>:d    r53.0<8;8,1>:d    2:w               {Compacted,@2}
        send.dc1 (16|M0)         r47      r41     null    0x0            0x04205EFE           {$9} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r49      r43     null    0x0            0x04205EFE           {$10} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M0)         r55      r51     null    0x0            0x04205EFE           {@2,$11} // wr:2+0, rd:2; untyped surface read with x
        send.dc1 (16|M16)        r57      r53     null    0x0            0x04205EFE           {@1,$12} // wr:2+0, rd:2; untyped surface read with x
        sync.nop                             null                             {Compacted,$11.dst}
        sync.nop                             null                             {Compacted,$6.src}
        add (16|M0)              r59.0<1>:f    r47.0<8;8,1>:f    r55.0<8;8,1>:f   {Compacted,$9.dst}
        sync.nop                             null                             {Compacted,$12.dst}
        sync.nop                             null                             {Compacted,$7.src}
        add (16|M16)             r61.0<1>:f    r49.0<8;8,1>:f    r57.0<8;8,1>:f   {Compacted,$10.dst}
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(~f0.0) if (32|M0)                           L1120                  L1120                
L1064:
        mul (16|M0)              acc0.0<1>:f   r59.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        mul (16|M16)             r65.0<1>:f    r61.0<8;8,1>:f    1024.0:f               {Compacted,@4}
        rnde (16|M0)             acc0.0<1>:f   acc0.0<8;8,1>:f                 
        rnde (16|M16)            r69.0<1>:f    r65.0<8;8,1>:f                   {Compacted,@2}
        mul (16|M0)              r59.0<1>:f    acc0.0<8;8,1>:f   9.765625e-04:f               {Compacted}
        mul (16|M16)             r61.0<1>:f    r69.0<8;8,1>:f    9.765625e-04:f               {Compacted,@2}
L1120:
        endif (32|M0)                        L1136                                
L1136:
        send.dc1 (16|M0)         null     r41     r59     0x80            0x04025EFE           {@3,$6} // wr:2+2, rd:0; untyped surface write with x
        send.dc1 (16|M16)        null     r43     r61     0x80            0x04025EFE           {@2,$7} // wr:2+2, rd:0; untyped surface write with x
L1168:
        endif (32|M0)                        L1184                                
L1184:
(W)     mov (1|M0)               r4.4<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       f0.0<1>:ud    0xFFFFFFFF:ud                             
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(W&f0.0) send.dc0 (8|M0)         r71      r3      null    0x0            0x0219E0FE           {$13} // wr:1h+0, rd:1; synchronized SLM fence
(W&f0.0) and (8|M0)              r72.0<1>:ud   r3.2<0;1,0>:ud    0x7F000000:ud              {$8.src}
(W)     mov (8|M0)               null<1>:ud    r71.0<8;8,1>:ud                  {$13.dst}
(W&f0.0) send.gtwy (1|M0)        null     r72     null    0x0            0x02000004           {@2,$8} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             null                            
(W)     mov (1|M0)               r4.3<1>:ud    f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M0)   (lt)f1.0   null<1>:d     r4.1<0;1,0>:ud    0x4:uw             
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r4.3<0;1,0>:ud                   {@2}
(W)     mov (1|M0)               r4.3<1>:ud    f1.0<0;1,0>:ud                   {Compacted}
(W)     cmp (16|M16)  (lt)f1.0   null<1>:d     r4.1<0;1,0>:ud    0x4:uw             
(W&~f0.0) mov (1|M0)             f1.0<1>:ud    r4.3<0;1,0>:ud                   {@2}
(W)     mov (1|M0)               f0.0<1>:ud    r4.4<0;1,0>:ud                  
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w        {@7}
(f1.0)  break (32|M0)                        L1536                  L1536                
L1464:
(W)     mov (1|M0)               r4.4<1>:ud    f0.0<0;1,0>:ud                   {Compacted}
(W)     mov (1|M0)               f0.0<1>:ud    0x0:ud                             
        cmp (32|M0)   (eq)f0.0   null<1>:uw    r2.0<0;1,0>:uw    r2.0<0;1,0>:uw  
(W&f0.0.any32h) mov (1|M0)       r4.1<1>:f     r4.2<0;1,0>:f                   
(W)     mov (1|M0)               f0.0<1>:ud    r4.4<0;1,0>:ud                   {@4}
L1536:
        while (32|M0)                        L752                                
L1552:
        endif (32|M0)                        L1568                                
L1568:
        cmp (16|M0)   (eq)f0.0   null<1>:w     r1.0<16;16,1>:w   0:w               {@7}
        cmp (16|M16)  (eq)f0.0   null<1>:w     r2.0<16;16,1>:w   0:w              
(W)     csel (4|M0)   (eq)f0.0   r1.0<1>:w     r1.0<4;1>:w       r1.0<4;1>:w       r1.0<1>:w       
(f0.0)  if (32|M0)                           L1680                  L1680                
L1632:
(W)     mov (1|M0)               r73.0<1>:f    0.0:f                               {Compacted}
(W)     shl (1|M0)               r74.0<1>:d    r3.1<0;1,0>:d     2:w               {Compacted}
(W)     send.dc0 (1|M0)          r8       r73     null    0x0            0x021108FE           {@2,$14} // wr:1+0, rd:1; byte gathering read 32b
(W)     send.dc0 (1|M0)          null     r74     r8      0x40            0x02030801           {@1,$14} // wr:1+1, rd:0; byte scattering write 32b
L1680:
        endif (32|M0)                        L1696                                
L1696:
(W)     mov (8|M0)               r127.0<1>:f   r3.0<8;8,1>:f                    {Compacted}
(W)     send.dc0 (8|M0)          r75      r3      null    0x0            0x0219E000           {$15} // wr:1h+0, rd:1; synchronized global fence flushing
(W)     mov (8|M0)               null<1>:ud    r75.0<8;8,1>:ud                  {$15.dst}
(W)     send.dc0 (8|M0)          r76      r3      null    0x0            0x0219E0FE           {$0} // wr:1h+0, rd:1; synchronized SLM fence
(W)     mov (8|M0)               null<1>:ud    r76.0<8;8,1>:ud                  {$0.dst}
(W)     mov (16|M0)              acc0.0<1>:f   0.0:f                              
(W)     send.ts (1|M0)           null     r127    null    0x0            0x02000010           {EOT,@1} // wr:1+0, rd:0; end of thread
L1800:
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
