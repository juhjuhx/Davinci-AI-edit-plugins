!ifndef VERSION
  !define VERSION "4.0"
!endif
!ifndef OUTDIR
  !define OUTDIR "dist"
!endif

Name "Smart A-Roll v${VERSION}"
OutFile "${OUTDIR}\SmartARoll_v${VERSION}_Setup.exe"
InstallDir "$PROGRAMFILES\SmartARoll"
RequestExecutionLevel admin

Page directory
Page instfiles

Section "Install"
  SetOutPath $INSTDIR
  
  File "dist\SmartARoll_v4.0\SmartARoll.exe"
  File "dist\SmartARoll_v4.0\config.json"
  File "dist\SmartARoll_v4.0\launch.bat"
  File "dist\SmartARoll_v4.0\apply_to_resolve.py"
  
  CreateDirectory "$INSTDIR\results"
  CreateDirectory "$INSTDIR\logs"
  
  IfFileExists "dist\SmartARoll_v4.0\ffmpeg\ffmpeg.exe" 0 +3
    CreateDirectory "$INSTDIR\ffmpeg"
    File /nonfatal "dist\SmartARoll_v4.0\ffmpeg\ffmpeg.exe"
    File /nonfatal "dist\SmartARoll_v4.0\ffmpeg\ffprobe.exe"
  
  CreateShortCut "$DESKTOP\Smart A-Roll.lnk" "$INSTDIR\launch.bat" "" "" 0
  CreateDirectory "$SMPROGRAMS\Smart A-Roll"
  CreateShortCut "$SMPROGRAMS\Smart A-Roll\Smart A-Roll.lnk" "$INSTDIR\launch.bat"
  CreateShortCut "$SMPROGRAMS\Smart A-Roll\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SmartARoll" \
    "DisplayName" "Smart A-Roll v${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SmartARoll" \
    "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SmartARoll" \
    "InstallLocation" "$INSTDIR"
SectionEnd

Section "Uninstall"
  Delete "$INSTDIR\SmartARoll.exe"
  Delete "$INSTDIR\config.json"
  Delete "$INSTDIR\launch.bat"
  Delete "$INSTDIR\apply_to_resolve.py"
  Delete "$INSTDIR\uninstall.exe"
  RMDir /r "$INSTDIR\ffmpeg"
  RMDir /r "$INSTDIR\results"
  RMDir /r "$INSTDIR\logs"
  RMDir "$INSTDIR"
  
  Delete "$DESKTOP\Smart A-Roll.lnk"
  RMDir /r "$SMPROGRAMS\Smart A-Roll"
  
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SmartARoll"
SectionEnd
