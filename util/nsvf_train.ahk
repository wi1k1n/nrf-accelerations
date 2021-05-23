#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
; SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Reconfigure train and debug ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
^!+q::
; Run reconfiguration script and then train
ReconfigureTrain()
Train(1500)
return

;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Reconfigure train ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;
^!+w::
; Only run reconfiguration script
ReconfigureTrain()
return

;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Reconfigure render ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;
^!+e::
; Only run reconfiguration script
ReconfigureRender()
return

ReconfigureTrain() {
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
ReconfigureRender() {
    SetWorkingDir, D:\edu\UniBonn\Study\thesis\codes\NSVF
    RunWait, C:\Python37\python.exe "D:/edu/UniBonn/Study/thesis/codes/NSVF/util/reconfigure_render.py"
    return
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Download backup from overleaf ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
^!+o::
;;; Look for 'menu' button
CoordMode Pixel
ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, *10 img\overleaf_menu.png

if (ErrorLevel) { ; 'menu' button not found
    Msgbox, Menu button not found!
    return
}

CoordMode, Mouse, Screen
MouseClick, Left, FoundX+45, FoundY+20
sleep, 500


;;; Look for 'source' button
CoordMode Pixel
ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, *10 img\overleaf_source.png

if (ErrorLevel) { ; 'source' button not found
    Msgbox, Source button not found!
    return
}


CoordMode, Mouse, Screen
MouseClick, Left, FoundX+38, FoundY+44

; Wait for save as dialog to appear
if (w8forWin2Appear("Save as", 7000) = 0) {
    Msgbox, Save as window not found!
    return
}

Sleep, 1000

FormatTime, TimeString,, yyyyMMdd_hhmmss
archiveName := % "overleaf_" . TimeString
Send, %archiveName%

Sleep, 250

Send !d

Sleep, 250

archiveDir := "D:\edu\UniBonn\Study\thesis\codes\backup\overleaf"

Send, %archiveDir%
Send, {Enter}

Sleep, 1000

Send, {Enter}

archivePath = % archiveDir . "\" . archiveName . ".zip"

; Wait for file to be downloaded
i := 0
while (i < 7000) {
    if FileExist(archivePath) {
        break
    }
    sleep, 100
    i := i + 100
}
if (i >= 7000) {
    Msgbox, Could not find downloaded archive
    return
}

Run, % "C:\Program Files\WinRAR\WinRAR.exe " . archivePath
Sleep, 1000

Send, !e
Sleep, 1000

Send, D:\edu\UniBonn\Study\thesis\codes\NSVF\writings\overleaf
Sleep, 1000

Send, {enter}

if w8forWin2Appear("Confirm file replace", 3000) {
    Send, {tab}
    Sleep, 50
    Send, {enter}
    Sleep, 3000
}

winget, winid, ID, A
WinGetTitle, wint, ahk_id %winid%
winrarTitle := % archiveName . ".zip - WinRAR"
if (wint = winrarTitle) {
    Send, !{F4}
}

return


; Waits until the window with title appears
; returns 1 if found, 0 if timeouted
w8forWin2Appear(title, timeout := 1000) {
    i := 0
    while (i < timeout) {
        winget, winid, ID, A
        WinGetTitle, wint, ahk_id %winid%
        winGetClass, winc, ahk_id %winid%
        if (wint = title) {
            break
        }
        sleep, 100
        i := i + 100
    }
    return i < timeout
}