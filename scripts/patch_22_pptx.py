# -*- coding: utf-8 -*-
"""
Script chỉnh sửa trực tiếp file `22.pptx`:
1. Render toàn bộ công thức toán trong Phần II (Slides 10, 11, 12, 13) và Phần IV (Slides 27, 28)
   bằng Computer Modern LaTeX (màu #29473D, nền trong suốt, 300 DPI) giống hệt định dạng
   công thức của Phần III (Slides 17, 18, 19, 20, 21) và Slide 6.
2. Cập nhật toàn bộ 36 Speaker Notes (`ppt/notesSlides/notesSlide1..36.xml`) để dòng đầu tiên
   luôn ghi rõ:
   `Người trình bày: Nguyễn Việt Khoa` (22 slides: 01-04, 06-08, 10-16, 27-31, 34-36)
   hoặc
   `Người trình bày: Nguyễn Đăng Khôi` (14 slides: 05, 09, 17-26, 32, 33)
   kèm nội dung thuyết minh hoàn chỉnh (<= 500 từ/slide).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.generate_canva_36_speaker_notes import CANVA_36_SLIDES

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("p", P_NS)


def render_latex_block(
    lines: List[Tuple[float, str, int]],
    out_path: Path,
    figsize: Tuple[float, float] = (12.0, 3.0),
    align: str = "center",
) -> Tuple[int, int]:
    """
    Render một khối công thức LaTeX Computer Modern màu #29473D trên nền trong suốt,
    cắt sát vùng bounding box thực tế của các ký tự (kèm lề nhỏ), trả về (width_px, height_px).
    """
    fig = plt.figure(figsize=figsize, dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    x_pos = 0.5 if align == "center" else 0.02
    for y_pos, tex, fsize in lines:
        ax.text(
            x_pos,
            y_pos,
            tex,
            ha=align,
            va="center",
            fontsize=fsize,
            color="#29473D",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".raw.png")
    fig.savefig(tmp_path, transparent=True, dpi=300)
    plt.close(fig)

    # Cắt tự động vùng alpha > 0 kèm padding 18px để không bao giờ mất dấu mũ/chỉ số
    im = Image.open(tmp_path).convert("RGBA")
    arr = np.array(im)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 4)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(f"Rendered empty formula image for {out_path}")
    pad_x, pad_y = 24, 18
    x0 = max(0, int(xs.min()) - pad_x)
    x1 = min(im.width, int(xs.max()) + pad_x + 1)
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(im.height, int(ys.max()) + pad_y + 1)
    cropped = im.crop((x0, y0, x1, y1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out_path)
    tmp_path.unlink(missing_ok=True)
    return cropped.size


def fit_box_keep_aspect(
    img_w: int,
    img_h: int,
    box_x: int,
    box_y: int,
    box_cx: int,
    box_cy: int,
    h_align: str = "center",
    v_align: str = "center",
) -> Tuple[int, int, int, int]:
    """
    Tính tọa độ (x, y, cx, cy) theo EMU giữ nguyên 100% tỷ lệ khung hình (aspect ratio)
    của ảnh công thức bên trong hộp giới hạn (box_x, box_y, box_cx, box_cy).
    """
    aspect = img_w / float(img_h)
    box_aspect = box_cx / float(box_cy)
    if aspect >= box_aspect:
        cx = box_cx
        cy = int(round(cx / aspect))
    else:
        cy = box_cy
        cx = int(round(cy * aspect))

    if h_align == "center":
        x = box_x + (box_cx - cx) // 2
    elif h_align == "left":
        x = box_x
    else:
        x = box_x + (box_cx - cx)

    if v_align == "center":
        y = box_y + (box_cy - cy) // 2
    elif v_align == "top":
        y = box_y
    else:
        y = box_y + (box_cy - cy)

    return x, y, cx, cy


def make_formula_sp_element(
    shape_id: int,
    name: str,
    descr: str,
    r_id: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
) -> ET.Element:
    """
    Tạo phần tử XML `<p:sp>` chứa `<a:blipFill>` giống hệt cấu trúc chèn ảnh LaTeX
    của Canva trên Slide 17, 18, 19 trong file `22.pptx`.
    """
    xml_str = (
        f'<p:sp xmlns:p="{P_NS}" xmlns:a="{A_NS}" xmlns:r="{R_NS}">'
        f'<p:nvSpPr>'
        f'<p:cNvPr id="{shape_id}" name="{name}" descr="{descr}"/>'
        f'<p:cNvSpPr/>'
        f'<p:nvPr/>'
        f'</p:nvSpPr>'
        f'<p:spPr>'
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:custGeom>'
        f'<a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/>'
        f'<a:rect l="l" t="t" r="r" b="b"/>'
        f'<a:pathLst>'
        f'<a:path w="{cx}" h="{cy}">'
        f'<a:moveTo><a:pt x="0" y="0"/></a:moveTo>'
        f'<a:lnTo><a:pt x="{cx}" y="0"/></a:lnTo>'
        f'<a:lnTo><a:pt x="{cx}" y="{cy}"/></a:lnTo>'
        f'<a:lnTo><a:pt x="0" y="{cy}"/></a:lnTo>'
        f'<a:lnTo><a:pt x="0" y="0"/></a:lnTo>'
        f'<a:close/>'
        f'</a:path>'
        f'</a:pathLst>'
        f'</a:custGeom>'
        f'<a:blipFill>'
        f'<a:blip r:embed="{r_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</a:blipFill>'
        f'</p:spPr>'
        f'</p:sp>'
    )
    return ET.fromstring(xml_str)


def add_hyperlink_relationship(rels_xml_bytes: bytes, url: str) -> Tuple[bytes, str]:
    ET.register_namespace("", REL_NS)
    root = ET.fromstring(rels_xml_bytes)
    existing_ids = [el.attrib.get("Id", "") for el in root]
    max_num = 0
    for rid in existing_ids:
        if rid.startswith("rId") and rid[3:].isdigit():
            max_num = max(max_num, int(rid[3:]))
    new_rid = f"rId{max_num + 1}"
    rel_el = ET.SubElement(root, f"{{{REL_NS}}}Relationship")
    rel_el.set("Id", new_rid)
    rel_el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink")
    rel_el.set("Target", url)
    rel_el.set("TargetMode", "External")
    out_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return out_bytes, new_rid


def attach_hyperlinks_to_slide(slide_xml_bytes: bytes, rels_xml_bytes: bytes, url_map: Dict[str, str]) -> Tuple[bytes, bytes]:
    root = ET.fromstring(slide_xml_bytes)
    for r_el in root.findall(f".//{{{A_NS}}}r"):
        t_el = r_el.find(f"{{{A_NS}}}t")
        if t_el is None or not t_el.text:
            continue
        for key, url in url_map.items():
            if key in t_el.text:
                rels_xml_bytes, rid = add_hyperlink_relationship(rels_xml_bytes, url)
                rpr = r_el.find(f"{{{A_NS}}}rPr")
                if rpr is None:
                    rpr = ET.Element(f"{{{A_NS}}}rPr")
                    r_el.insert(0, rpr)
                if rpr.find(f"{{{A_NS}}}hlinkClick") is None:
                    ET.SubElement(rpr, f"{{{A_NS}}}hlinkClick", {f"{{{R_NS}}}id": rid})
                break
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), rels_xml_bytes


def add_image_relationship(rels_xml_bytes: bytes, media_filename: str) -> Tuple[bytes, str]:
    """
    Thêm `<Relationship>` trỏ tới `../media/<media_filename>` vào `slideX.xml.rels`
    và trả về `(updated_rels_xml_bytes, new_rId)`.
    """
    ET.register_namespace("", REL_NS)
    root = ET.fromstring(rels_xml_bytes)
    existing_ids = [el.attrib.get("Id", "") for el in root]
    max_num = 0
    for rid in existing_ids:
        if rid.startswith("rId") and rid[3:].isdigit():
            max_num = max(max_num, int(rid[3:]))
    new_rid = f"rId{max_num + 1}"
    rel_el = ET.SubElement(root, f"{{{REL_NS}}}Relationship")
    rel_el.set("Id", new_rid)
    rel_el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel_el.set("Target", f"../media/{media_filename}")
    out_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return out_bytes, new_rid


def shape_text_joined(sp: ET.Element) -> str:
    return " ".join(t.text.strip() for t in sp.findall(f".//{{{A_NS}}}t") if t.text and t.text.strip())


def remove_shapes_matching(sp_tree: ET.Element, substrings: List[str]) -> int:
    removed = 0
    for child in list(sp_tree):
        txt = shape_text_joined(child)
        if any(sub in txt for sub in substrings):
            sp_tree.remove(child)
            removed += 1
    return removed


def build_all_formula_images(media_dir: Path) -> Dict[str, Tuple[Path, Tuple[int, int]]]:
    """
    Render toàn bộ 8 ảnh công thức LaTeX cho Phần II (Slides 10, 11, 12, 13)
    và Phần IV (Slides 27, 28).
    """
    specs = {
        # Slide 10 - Mô hình dữ liệu
        "eq_s10_model.png": (
            (11.0, 2.3),
            "left",
            [
                (0.74, r"$y_{t,k} = x_{t,k}^{\top} \theta_t^{*} + \varepsilon_{t,k} \qquad (7)$", 28),
                (0.23, r"$x_{t,k} \sim \mathcal{N}(0, I_d), \qquad \varepsilon_{t,k} \sim \mathcal{N}(0, \sigma^2)$", 27),
            ],
        ),
        # Slide 10 - Hàm mất mát
        "eq_s10_loss.png": (
            (8.5, 2.4),
            "left",
            [
                (0.75, r"$\ell(\theta; (x, y)) = \frac{1}{2}\left(y - x^{\top} \theta\right)^2$", 26),
                (0.23, r"$\bar{\ell}_t(\theta) - \bar{\ell}_t(\theta_t^{*}) = \frac{1}{2}\left\|\theta - \theta_t^{*}\right\|_2^2$", 26),
            ],
        ),
        # Slide 11 - Phương trình SDE (Eq. 8)
        "eq_s11_sde.png": (
            (15.5, 3.0),
            "center",
            [
                (0.77, r"$\tau = \varepsilon t, \qquad \eta_t = \varepsilon \zeta(\varepsilon t), \qquad B_t = \lceil \varepsilon \nu(\varepsilon t) \rceil, \qquad Y(\varepsilon t) = \theta_t^{*}$", 26),
                (0.27, r"$\mathrm{d}X(\tau) = -\left(\zeta(\tau)X(\tau) + Y'(\tau)\right)\mathrm{d}\tau + \frac{\zeta(\tau)}{\sqrt{\nu(\tau)}}\left[\left(\|X(\tau)\|_2^2 + \sigma^2\right)I_d + X(\tau)X(\tau)^{\top}\right]^{1/2}\mathrm{d}W(\tau) \quad (8)$", 25),
            ],
        ),
        # Slide 12 - Hệ phương trình Moment ODE (Eqs. 10-11)
        "eq_s12_ode.png": (
            (15.0, 3.0),
            "center",
            [
                (0.76, r"$m_\tau = \mathbb{E}[X(\tau)], \quad v_\tau = \mathbb{E}\left[\|X(\tau)\|_2^2\right], \qquad m_\tau' = -\zeta(\tau)m_\tau - Y'(\tau) \qquad (10)$", 26),
                (0.25, r"$v_\tau' = \left(\frac{d+1}{\nu(\tau)}\zeta(\tau)^2 - 2\zeta(\tau)\right)v_\tau + \frac{\zeta(\tau)^2}{\nu(\tau)}\sigma^2 d - 2m_\tau^{\top} Y'(\tau) \qquad (11)$", 26),
            ],
        ),
        # Slide 12 - Quá trình bao trên Scalar Envelope (Eq. 12)
        "eq_s12_env.png": (
            (8.5, 1.6),
            "left",
            [
                (0.50, r"$\tilde{v}_\tau' = \left(\frac{d+1}{\nu}\zeta^2 - 2\zeta\right)\tilde{v}_\tau + \frac{\zeta^2}{\nu}\sigma^2 d + 2\|Y'\|_2\sqrt{\tilde{v}_\tau} \quad (12)$", 23),
            ],
        ),
        # Slide 12 - Nghiệm điều khiển tối ưu HJB (Eq. 13)
        "eq_s12_ctrl.png": (
            (7.5, 1.6),
            "left",
            [
                (0.50, r"$\zeta^{*}(\tau) = \min\left\{1, \frac{\nu(\tau)\tilde{v}_\tau}{(d+1)\tilde{v}_\tau + \sigma^2 d}\right\} \quad (13)$", 25),
            ],
        ),
        # Slide 13 - Algorithm 1 (Rate & Euler step)
        "eq_s13_alg1.png": (
            (14.5, 2.9),
            "center",
            [
                (0.75, r"$r = \min\left\{\varepsilon, \frac{v B_t}{(d+1)v + \sigma^2 d}\right\}$", 27),
                (0.24, r"$v \leftarrow v + \kappa\left[\left(\frac{d+1}{B_t}r^2 - 2r\right)v + \frac{\sigma^2 d}{B_t}r^2 + 2\gamma_{t-1}\sqrt{v}\right]$", 27),
            ],
        ),
        # Slide 27 - Hàm không lồi & chuẩn gradient (Eqs. 22-23)
        "eq_s27_nonconvex.png": (
            (13.5, 3.0),
            "center",
            [
                (0.75, r"$\mathrm{Reg}(T) = \sum_{t=1}^T \left\|\nabla F_t(\theta_t)\right\|_2^2 \qquad (22)$", 28),
                (0.23, r"$\gamma_t = \sup_{\theta \in \mathbb{R}^p} \left|F_t(\theta) - F_{t+1}(\theta)\right| \qquad (23)$", 28),
            ],
        ),
        # Slide 28 - Theorem 5.1 & lịch tốc độ học đóng (Eqs. 24-25)
        "eq_s28_thm51.png": (
            (15.5, 4.5),
            "center",
            [
                (0.86, r"$a_t = 2\eta_t - L\eta_t^2 > 0, \qquad \mathbb{E}[\mathrm{Reg}(T)] \leq 2\mathbb{E}\left[\frac{F_1(\theta_1)}{a_1} + \sum_{t=2}^T F_t(\theta_t)\left(\frac{1}{a_t} - \frac{1}{a_{t-1}}\right)\right] + \sum_{t=1}^T \mathbb{E}\left[\frac{L\sigma^2\eta_t^2/B_t + 2\gamma_t}{a_t}\right] \quad (24)$", 22),
                (0.50, r"$\eta_t^{*} = \mathrm{arg}\,\min_{0 \leq \eta \leq 1/L} \left\{ \frac{2F_t(\theta_t) + 2\gamma_t}{2\eta - L\eta^2} + \frac{L\sigma^2}{B_t}\frac{\eta^2}{2\eta - L\eta^2} \right\} \qquad (25)$", 25),
                (0.14, r"$b_t = L\left(\gamma_t + F_t(\theta_t)\right), \qquad \eta_t^{*} = \frac{B_t}{L\sigma^2}\left(\sqrt{b_t^2 + \frac{2\sigma^2 b_t}{B_t}} - b_t\right)$", 25),
            ],
        ),
    }

    rendered: Dict[str, Tuple[Path, Tuple[int, int]]] = {}
    for fname, (fsize, align, lines) in specs.items():
        out_p = media_dir / fname
        dims = render_latex_block(lines=lines, out_path=out_p, figsize=fsize, align=align)
        rendered[fname] = (out_p, dims)
        print(f"[Rendered] {fname}: {dims[0]}x{dims[1]} px")
    return rendered


def patch_notes_xml(notes_xml_bytes: bytes, presenter_line: str, body_text: str) -> bytes:
    """
    Cập nhật phần tử `<p:ph type="body" ...>` trong `notesSlideX.xml` gồm 2 đoạn `<a:p>`:
    - Đoạn 1 (in đậm): `Người trình bày: Nguyễn Việt Khoa` (hoặc `Người trình bày: Nguyễn Đăng Khôi`)
    - Đoạn 2: Nội dung thuyết minh chi tiết của trang slide đó.
    """
    root = ET.fromstring(notes_xml_bytes)
    for sp in root.findall(f".//{{{P_NS}}}sp"):
        ph = sp.find(f".//{{{P_NS}}}ph")
        if ph is not None and ph.attrib.get("type") == "body":
            tx_body = sp.find(f"{{{P_NS}}}txBody")
            if tx_body is None:
                continue
            # Giữ lại bodyPr và lstStyle, xóa các <a:p> cũ
            for p_el in list(tx_body.findall(f"{{{A_NS}}}p")):
                tx_body.remove(p_el)

            # Đoạn 1: Người trình bày: ...
            p1 = ET.SubElement(tx_body, f"{{{A_NS}}}p")
            r1 = ET.SubElement(p1, f"{{{A_NS}}}r")
            rpr1 = ET.SubElement(r1, f"{{{A_NS}}}rPr", {"lang": "vi-VN", "b": "1", "dirty": "0"})
            t1 = ET.SubElement(r1, f"{{{A_NS}}}t")
            t1.text = presenter_line

            # Đoạn 2: Nội dung thuyết minh
            p2 = ET.SubElement(tx_body, f"{{{A_NS}}}p")
            r2 = ET.SubElement(p2, f"{{{A_NS}}}r")
            rpr2 = ET.SubElement(r2, f"{{{A_NS}}}rPr", {"lang": "vi-VN", "dirty": "0"})
            t2 = ET.SubElement(r2, f"{{{A_NS}}}t")
            t2.text = body_text
            break

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    src_pptx = PROJECT_ROOT / "slides" / "22_original_backup.pptx"
    if not src_pptx.exists():
        raise FileNotFoundError(f"Backup file not found: {src_pptx}")

    formula_dir = PROJECT_ROOT / "slides" / "rendered_formulas"
    rendered = build_all_formula_images(formula_dir)

    # Đọc toàn bộ entries từ file gốc
    with zipfile.ZipFile(src_pptx, "r") as zin:
        entries: Dict[str, bytes] = {name: zin.read(name) for name in zin.namelist()}

    # --- 1. Patch Slide 10 ---
    s10_root = ET.fromstring(entries["ppt/slides/slide10.xml"])
    s10_tree = s10_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(s10_tree, ["yₜ,ₖ", "Population excess loss"])
    entries["ppt/slides/_rels/slide10.xml.rels"], rid_10a = add_image_relationship(
        entries["ppt/slides/_rels/slide10.xml.rels"], "eq_s10_model.png"
    )
    entries["ppt/slides/_rels/slide10.xml.rels"], rid_10b = add_image_relationship(
        entries["ppt/slides/_rels/slide10.xml.rels"], "eq_s10_loss.png"
    )
    w10a, h10a = rendered["eq_s10_model.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w10a, h10a, 952500, 2750000, 12500000, 1550000, h_align="left", v_align="center")
    s10_tree.append(make_formula_sp_element(901, "Formula 10A", "Linear model Eq. (7)", rid_10a, x, y, cx, cy))

    w10b, h10b = rendered["eq_s10_loss.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w10b, h10b, 952500, 5280000, 7600000, 1650000, h_align="left", v_align="top")
    s10_tree.append(make_formula_sp_element(902, "Formula 10B", "Squared loss and excess loss", rid_10b, x, y, cx, cy))
    entries["ppt/slides/slide10.xml"] = ET.tostring(s10_root, encoding="utf-8", xml_declaration=True)

    # --- 2. Patch Slide 11 ---
    s11_root = ET.fromstring(entries["ppt/slides/slide11.xml"])
    s11_tree = s11_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(s11_tree, ["τ = εt", "dX(τ)"])
    entries["ppt/slides/_rels/slide11.xml.rels"], rid_11 = add_image_relationship(
        entries["ppt/slides/_rels/slide11.xml.rels"], "eq_s11_sde.png"
    )
    w11, h11 = rendered["eq_s11_sde.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w11, h11, 952500, 2100000, 16383000, 2550000, h_align="center", v_align="center")
    s11_tree.append(make_formula_sp_element(903, "Formula 11", "Continuous-time SDE Eq. (8)", rid_11, x, y, cx, cy))
    entries["ppt/slides/slide11.xml"] = ET.tostring(s11_root, encoding="utf-8", xml_declaration=True)

    # --- 3. Patch Slide 12 ---
    s12_root = ET.fromstring(entries["ppt/slides/slide12.xml"])
    s12_tree = s12_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(s12_tree, ["m(τ)=E[X(τ)]", "m′ = −ζm"])
    # Với hộp bên phải под "Điều khiển tốc độ học", xóa dòng công thức text đầu tiên và chỉ giữ câu ghi chú ở dưới
    for child in list(s12_tree):
        txt = shape_text_joined(child)
        if "ζ* = min" in txt:
            # Sửa hộp văn bản này thành chỉ chứa câu nhận xét Theorem 3.3 và dời xuống y=6350000
            xfrm = child.find(f".//{{{A_NS}}}xfrm")
            if xfrm is not None:
                off = xfrm.find(f"{{{A_NS}}}off")
                ext = xfrm.find(f"{{{A_NS}}}ext")
                if off is not None:
                    off.set("y", "6350000")
                if ext is not None:
                    ext.set("cy", "680000")
            tx_body = child.find(f"{{{P_NS}}}txBody")
            if tx_body is not None:
                ps = tx_body.findall(f"{{{A_NS}}}p")
                if len(ps) >= 2:
                    tx_body.remove(ps[0])
    entries["ppt/slides/_rels/slide12.xml.rels"], rid_12a = add_image_relationship(
        entries["ppt/slides/_rels/slide12.xml.rels"], "eq_s12_ode.png"
    )
    entries["ppt/slides/_rels/slide12.xml.rels"], rid_12b = add_image_relationship(
        entries["ppt/slides/_rels/slide12.xml.rels"], "eq_s12_ctrl.png"
    )
    entries["ppt/slides/_rels/slide12.xml.rels"], rid_12c = add_image_relationship(
        entries["ppt/slides/_rels/slide12.xml.rels"], "eq_s12_env.png"
    )
    w12a, h12a = rendered["eq_s12_ode.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w12a, h12a, 952500, 1980000, 16383000, 2250000, h_align="center", v_align="center")
    s12_tree.append(make_formula_sp_element(904, "Formula 12A", "Moment ODEs Eqs. (10)-(11)", rid_12a, x, y, cx, cy))

    w12c, h12c = rendered["eq_s12_env.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w12c, h12c, 952500, 5820000, 7810500, 1200000, h_align="left", v_align="center")
    s12_tree.append(make_formula_sp_element(909, "Formula 12C", "Scalar envelope ODE Eq. (12)", rid_12c, x, y, cx, cy))

    w12b, h12b = rendered["eq_s12_ctrl.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w12b, h12b, 9239250, 4960000, 8001000, 1250000, h_align="left", v_align="center")
    s12_tree.append(make_formula_sp_element(905, "Formula 12B", "Optimal control Eq. (13)", rid_12b, x, y, cx, cy))
    entries["ppt/slides/slide12.xml"] = ET.tostring(s12_root, encoding="utf-8", xml_declaration=True)

    # --- 4. Patch Slide 13 ---
    s13_root = ET.fromstring(entries["ppt/slides/slide13.xml"])
    s13_tree = s13_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(s13_tree, ["r = min{ε"])
    entries["ppt/slides/_rels/slide13.xml.rels"], rid_13 = add_image_relationship(
        entries["ppt/slides/_rels/slide13.xml.rels"], "eq_s13_alg1.png"
    )
    w13, h13 = rendered["eq_s13_alg1.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w13, h13, 952500, 4180000, 16383000, 2000000, h_align="center", v_align="center")
    s13_tree.append(make_formula_sp_element(906, "Formula 13", "Algorithm 1 Euler integration", rid_13, x, y, cx, cy))
    entries["ppt/slides/slide13.xml"] = ET.tostring(s13_root, encoding="utf-8", xml_declaration=True)

    # --- 5. Patch Slide 27 ---
    s27_root = ET.fromstring(entries["ppt/slides/slide27.xml"])
    s27_tree = s27_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(s27_tree, ["Reg(T) = Σ", "γₜ = sup"])
    entries["ppt/slides/_rels/slide27.xml.rels"], rid_27 = add_image_relationship(
        entries["ppt/slides/_rels/slide27.xml.rels"], "eq_s27_nonconvex.png"
    )
    w27, h27 = rendered["eq_s27_nonconvex.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w27, h27, 952500, 2800000, 16383000, 2500000, h_align="center", v_align="center")
    s27_tree.append(make_formula_sp_element(907, "Formula 27", "Non-convex gradient-norm regret Eqs. (22)-(23)", rid_27, x, y, cx, cy))
    entries["ppt/slides/slide27.xml"] = ET.tostring(s27_root, encoding="utf-8", xml_declaration=True)

    # --- 6. Patch Slide 28 ---
    s28_root = ET.fromstring(entries["ppt/slides/slide28.xml"])
    s28_tree = s28_root.find(f".//{{{P_NS}}}spTree")
    remove_shapes_matching(
        s28_tree,
        [
            "Cận (24) gồm biến thiên",
            "Eq. (25): tối thiểu hóa",
            "[2Fₜ(θₜ)+2γₜ]",
            "Nghiệm đóng với bₜ",
        ],
    )
    entries["ppt/slides/_rels/slide28.xml.rels"], rid_28 = add_image_relationship(
        entries["ppt/slides/_rels/slide28.xml.rels"], "eq_s28_thm51.png"
    )
    w28, h28 = rendered["eq_s28_thm51.png"][1]
    x, y, cx, cy = fit_box_keep_aspect(w28, h28, 952500, 2750000, 16383000, 5000000, h_align="center", v_align="center")
    s28_tree.append(make_formula_sp_element(908, "Formula 28", "Theorem 5.1 Eqs. (24)-(25)", rid_28, x, y, cx, cy))
    entries["ppt/slides/slide28.xml"] = ET.tostring(s28_root, encoding="utf-8", xml_declaration=True)

    # Đưa 8 file ảnh công thức vào `ppt/media/`
    for fname, (fpath, _) in rendered.items():
        entries[f"ppt/media/{fname}"] = fpath.read_bytes()

    # --- 7. Patch toàn bộ 36 Speaker Notes (`ppt/notesSlides/notesSlide1..36.xml`) ---
    for s_num, section, owner, title, raw_note in CANVA_36_SLIDES:
        presenter_name = "Nguyễn Việt Khoa" if "Khoa" in owner else "Nguyễn Đăng Khôi"
        presenter_line = f"Người trình bày: {presenter_name}"
        notes_key = f"ppt/notesSlides/notesSlide{s_num}.xml"
        if notes_key in entries:
            entries[notes_key] = patch_notes_xml(entries[notes_key], presenter_line, raw_note)

    # Lưu bản hoàn chỉnh vào thư mục PROJECT (`22.pptx` và `slides/22.pptx`) và `Downloads`
    out_targets = [
        PROJECT_ROOT / "22.pptx",
        PROJECT_ROOT / "slides" / "22.pptx",
        Path(r"C:\Users\Thinkpad X1\Downloads\22_final.pptx"),
    ]
    for target in out_targets:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in entries.items():
                zout.writestr(k, v)
        print(f"[OK] Wrote patched presentation ({target.stat().st_size / 1024 / 1024:.2f} MB) -> {target}")

    # Thử ghi đè trực tiếp `C:\Users\Thinkpad X1\Downloads\22.pptx` nếu không bị khóa bởi PowerPoint
    dl_22 = Path(r"C:\Users\Thinkpad X1\Downloads\22.pptx")
    try:
        with zipfile.ZipFile(dl_22, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for k, v in entries.items():
                zout.writestr(k, v)
        print(f"[OK] Directly updated: {dl_22}")
    except PermissionError:
        print(f"[INFO] {dl_22} is currently open in PowerPoint; saved ready-to-submit copies at {PROJECT_ROOT / '22.pptx'} and {dl_22.with_name('22_final.pptx')}")


if __name__ == "__main__":
    main()
