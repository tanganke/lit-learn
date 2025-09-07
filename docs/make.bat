@echo off
rem This batch file is used to build the documentation using Sphinx on Windows systems.

set SPHINXBUILD=python -m sphinx
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html
%SPHINXBUILD% -b latex %SOURCEDIR% %BUILDDIR%/latex
%SPHINXBUILD% -b man %SOURCEDIR% %BUILDDIR%/man

echo Documentation built successfully.