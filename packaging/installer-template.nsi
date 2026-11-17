Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"
Name "@NAME@"
OutFile "@OUTPUT@"
InstallDir "$LOCALAPPDATA\Programs\@NAME@"
InstallDirRegKey HKCU "Software\bandiaoseng\@NAME@" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 32
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "English"
Var TestInstall
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "This installer requires 64-bit Windows."
    Abort
  ${EndIf}
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "/TestInstall" $R1
  IfErrors +2 0
    StrCpy $TestInstall "1"
FunctionEnd
Section "Install"
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "@STAGE@\*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  StrCmp $TestInstall "1" done
  CreateDirectory "$SMPROGRAMS\@NAME@"
  CreateShortCut "$SMPROGRAMS\@NAME@\@NAME@.lnk" "$INSTDIR\@NAME@.exe"
  CreateShortCut "$DESKTOP\@NAME@.lnk" "$INSTDIR\@NAME@.exe"
  WriteRegStr HKCU "Software\bandiaoseng\@NAME@" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "DisplayName" "@NAME@"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "DisplayVersion" "0.1.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "DisplayIcon" "$INSTDIR\@NAME@.exe"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@" "NoRepair" 1
  done:
SectionEnd
Section "Uninstall"
  SetShellVarContext current
@DELETE@
  Delete "$INSTDIR\Uninstall.exe"
@REMOVE@
  RMDir "$INSTDIR"
  ReadRegStr $R0 HKCU "Software\bandiaoseng\@NAME@" "InstallDir"
  StrCmp $R0 $INSTDIR 0 doneUninstall
  Delete "$DESKTOP\@NAME@.lnk"
  Delete "$SMPROGRAMS\@NAME@\@NAME@.lnk"
  RMDir "$SMPROGRAMS\@NAME@"
  DeleteRegKey HKCU "Software\bandiaoseng\@NAME@"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\@NAME@"
  doneUninstall:
SectionEnd
