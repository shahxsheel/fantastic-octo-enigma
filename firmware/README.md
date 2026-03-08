# Firmware for the board

1. Install [Infineon ModusToolbox](https://www.infineon.com/design-resources/development-tools/sdk/modustoolbox-software). You will need to create a myInfineon account to do so.
2. Connect [PSOC Edge E84 AI Kit](https://www.infineon.com/evaluation-board/kit-pse84-ai) board to your computer with USB-C.
3. On the Eclipse editor, in the Quick Panel, click "Build Project".
4. Then click on "PSOC_EDGE_Hello_World.proj_cm33_ns Program (KitProg3_MiniProg4)".
5. Get location of the USB device.

```zsh
$ ls /dev/cu.* | grep usbmodem
```

6. See the output. (for example if the previous command's output was `/dev/cu.usbmodem2103`):

```zsh
$ screen /dev/cu.usbmodem2103 115200
```
