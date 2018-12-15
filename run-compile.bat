@echo off

set PATH=.;%PATH%;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\
call vcvarsall.bat

cl.exe /EHsc /nologo nn.cpp
del *.obj

pause
