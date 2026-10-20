rem %4 -- $CI_COMMIT_REF_NAME
set odls=%1
set gpu=%2

set fname=pkl_win.zip
echo %fname%

rem The open-source CPU lane has no suffix (perf_win.pkl, perf_win_<case>.pkl), so its
rem mask perf_win*.pkl also matches the _iter/_gpu references; cpu_excl drops them.
set suffix=
set cpu_excl=
if "%odls%"=="-a" set suffix=_iter
if "%gpu%"=="1" set suffix=_gpu
if "%suffix%"=="" set cpu_excl=-xr!perf_win_iter*.pkl -xr!perf_win_gpu*.pkl -xr!well_time_data_win_iter*.pkl -xr!well_time_data_win_gpu*.pkl

set pklname=perf_win%suffix%
set wellpklname=well_time_data_win%suffix%

rem # delete pkls from previous pipeline run
if exist %fname% del %fname%

if /I %UPLOAD_PKL% NEQ 1 exit

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%*.pkl %wellpklname%*.pkl %cpu_excl%

rem vtk references (THM_vs_nucleus_of_strain, displaced_fault_reactivation) are written
rem by save_vtk_ref during the same UPLOAD_PKL=1 run, so they belong in this archive.
rem Unlike the pkl names they carry no _iter/_gpu suffix: one reference serves
rem every lane. The masks are anchored on ref\ and used without -r, so the run
rem outputs under results\ are not swept in. 7z only warns when a mask matches
rem nothing, so a lane with no vtk references still succeeds.
"C:\Program files\7-Zip\7z.exe" a %fname% "*\ref\*\*.vtu" "*\*\ref\*\*.vtu"
