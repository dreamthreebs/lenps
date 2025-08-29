from pixell import enmap, enplot, utils
import numpy as np
import matplotlib.pyplot as plt

def interactive_show(m: enmap.enmap, title="Map", unit="µK", cmap="coolwarm", vmin=None, vmax=None, plt_show=False):
    decmin, ramin = m.box()[0] / utils.degree
    decmax, ramax = m.box()[1] / utils.degree
    extent = [ramin, ramax, decmin, decmax]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(m, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, fraction=0.046)
    cbar.set_label(unit)


    ax.set_title(title)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")

    def format_coord(x, y):
        pix = m.sky2pix([y * utils.degree, x * utils.degree])
        #pix = m.sky2pix([y, x])  # [Dec, RA]
        iy, ix = int(pix[0]), int(pix[1])
        if 0 <= iy < m.shape[-2] and 0 <= ix < m.shape[-1]:
            val = m[iy, ix]
            return f"RA={x:.3f}°, Dec={y:.3f}°, Value={val:.3f} {unit}"
        else:
            return f"RA={x:.3f}°, Dec={y:.3f}° (out of bounds)"

    #ax.format_coord = format_coord
    if plt_show:
        plt.show()

# 读入 Signal map 和 ivar map
signal_map = enmap.read_map("maps/f150_1_10_8_map.fits")
ivar_map   = enmap.read_map("maps/f150_1_10_8_ivar.fits")

# 计算 RMS 噪声: sigma = 1 / sqrt(ivar)，避免除0
rms_map = np.zeros_like(ivar_map)
mask_positive = ivar_map > 0
rms_map[mask_positive] = 1 / np.sqrt(ivar_map[mask_positive])
rms_map[~mask_positive] = np.nan  # 处理非法值

# 计算 SNR map
snr_map = np.zeros_like(signal_map)
snr_map[mask_positive] = signal_map[mask_positive] * np.sqrt(ivar_map[mask_positive])
snr_map[~mask_positive] = np.nan

interactive_show(signal_map, vmin=-200, vmax=200)
interactive_show(ivar_map)
interactive_show(snr_map, vmin=-5, vmax=5)
plt.show()