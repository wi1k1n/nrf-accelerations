^!+q::
; Run reconfiguration script
SetWorkingDir, D:\edu\UniBonn\Study\thesis\codes\NSVF
RunWait, C:\Python37\python.exe "D:/edu/UniBonn/Study/thesis/codes/NSVF/util/reconfigure_train.py"

; Find pycharm window
if WinExist("ahk_exe pycharm64.exe") {
    WinActivate, ahk_exe pycharm64.exe
	sleep, 1000
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