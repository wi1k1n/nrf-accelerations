^!+q::
; Run reconfiguration script and then train
Reconfigure()
Train(1500)
return

^!+w::
; Only run reconfiguration script
Reconfigure()
return

^!+e::
; Only run configuration
Train()
return

Reconfigure() {
    SetWorkingDir, D:\edu\UniBonn\Study\thesis\codes\NSVF
    RunWait, C:\Python37\python.exe "D:/edu/UniBonn/Study/thesis/codes/NSVF/util/reconfigure_train.py"
    return
}
Train(w8time := 0) {
    ; Find pycharm window
    if WinExist("ahk_exe pycharm64.exe") {
        WinActivate, ahk_exe pycharm64.exe
        sleep, w8time
        Send, {F5}
        sleep, 500
        Send, train
        sleep, 500
        Send, {Enter}
        ;MouseClick, Left, 578, 58
    }
    else {
        Msgbox, No PyCharm instance found!
    }
    return
}