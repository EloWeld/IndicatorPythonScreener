# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_dash_version.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sound_down.mp3', '.'),
        ('sound_up.mp3', '.'),
        ('last_signals.json', '.'),
        ('trading_signals.html', '.'),
        ('cached_prices.json', '.'),
        ('ind_config.json', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_dash_version',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_dash_version',
)
