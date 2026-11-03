"""
PIX Payment Service for Empório da Música.
Generates simulated QR Codes (SVG) and Copy-and-Paste (Copia e Cola) payloads.
"""

import hashlib
from typing import Optional


class PixService:
    @staticmethod
    def generate_pix_svg(
        amount: str = "0.00",
        name: str = "Empório da Música",
        txid: str = "EMP123456",
    ) -> str:
        """
        Generate a standalone SVG representation of a simulated PIX QR Code.
        Includes authentic QR matrix finder patterns, timing belts, alignment boxes,
        deterministic data payload modules, and official PIX branding overlay.
        """
        N = 25  # Version 2 QR matrix
        grid = [[0 for _ in range(N)] for _ in range(N)]
        reserved = [[False for _ in range(N)] for _ in range(N)]

        # 1. Place 3 Finder Patterns (7x7) + Separator margins
        def place_finder(r0: int, c0: int) -> None:
            for r in range(7):
                for c in range(7):
                    reserved[r0 + r][c0 + c] = True
                    if r in (0, 6) or c in (0, 6):
                        grid[r0 + r][c0 + c] = 1
                    elif 2 <= r <= 4 and 2 <= c <= 4:
                        grid[r0 + r][c0 + c] = 1
                    else:
                        grid[r0 + r][c0 + c] = 0
            for r in range(max(0, r0 - 1), min(N, r0 + 8)):
                for c in range(max(0, c0 - 1), min(N, c0 + 8)):
                    if not (r0 <= r < r0 + 7 and c0 <= c < c0 + 7):
                        reserved[r][c] = True
                        grid[r][c] = 0

        place_finder(0, 0)
        place_finder(0, N - 7)
        place_finder(N - 7, 0)

        # 2. Timing patterns on row 6 and col 6
        for i in range(8, N - 8):
            reserved[6][i] = True
            grid[6][i] = 1 if i % 2 == 0 else 0
            reserved[i][6] = True
            grid[i][6] = 1 if i % 2 == 0 else 0

        # 3. Alignment pattern at (18, 18)
        ar, ac = N - 7, N - 7
        for r in range(ar - 2, ar + 3):
            for c in range(ac - 2, ac + 3):
                reserved[r][c] = True
                if r in (ar - 2, ar + 2) or c in (ac - 2, ac + 2) or (r == ar and c == ac):
                    grid[r][c] = 1
                else:
                    grid[r][c] = 0

        # 4. Fill remaining data modules with deterministic hash of order details
        seed = f"EMPORIO_PIX_{name}_{amount}_{txid}"
        h = hashlib.sha256(seed.encode("utf-8")).digest()
        bit_idx = 0
        for r in range(N):
            for c in range(N):
                if not reserved[r][c]:
                    byte = h[(bit_idx // 8) % len(h)]
                    grid[r][c] = (byte >> (bit_idx % 8)) & 1
                    bit_idx += 1

        # 5. Build SVG
        margin = 18
        mod_size = 8
        width = N * mod_size + margin * 2
        height = width + 48

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" rx="16" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
        ]

        # Draw dark QR modules (skipping center 5x5 reserved for PIX icon)
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 1:
                    if 10 <= r <= 14 and 10 <= c <= 14:
                        continue
                    x = margin + c * mod_size
                    y = margin + r * mod_size
                    svg_parts.append(f'<rect x="{x}" y="{y}" width="{mod_size}" height="{mod_size}" fill="#0f172a"/>')

        # Center PIX Icon Box
        cx = margin + 10 * mod_size - 2
        cy = margin + 10 * mod_size - 2
        cw = 5 * mod_size + 4
        svg_parts.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{cw}" rx="8" fill="#ffffff" stroke="#32BCAD" stroke-width="2.5"/>')
        svg_parts.append(f'<path d="M{cx+cw/2-11} {cy+cw/2} L{cx+cw/2} {cy+cw/2-11} L{cx+cw/2+11} {cy+cw/2} L{cx+cw/2} {cy+cw/2+11} Z" fill="#32BCAD"/>')
        svg_parts.append(f'<text x="{cx+cw/2}" y="{cy+cw/2+4}" font-family="Arial, sans-serif" font-size="8" font-weight="900" fill="#ffffff" text-anchor="middle">PIX</text>')

        # Footer Badge
        fy = margin + N * mod_size + 22
        svg_parts.append(f'<text x="{width/2}" y="{fy}" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="bold" fill="#32BCAD" text-anchor="middle">✦ PAGAMENTO INSTANTÂNEO PIX ✦</text>')
        if amount and amount != "0.00":
            clean_amt = str(amount).replace("R$", "").strip()
            svg_parts.append(f'<text x="{width/2}" y="{fy+16}" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0f172a" text-anchor="middle">R$ {clean_amt}</text>')
        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    @staticmethod
    def generate_pix_copia_e_cola(
        amount: float = 0.0,
        txid: str = "EMP123456",
        recipient_key: str = "pagamentos@emporiodamusica.com.br",
    ) -> str:
        """
        Simulate an authentic EMV BR Code string for PIX Copia e Cola.
        """
        amt_str = f"{amount:.2f}"
        clean_txid = (txid or "EMP123456").replace("-", "")[:20]
        # Standard EMV QRCPS format snippet
        return (
            f"00020126580014br.gov.bcb.pix0136{recipient_key}"
            f"520400005303986540{amt_str}5802BR5917Emporio da Musica"
            f"6012Campo Grande62070503{clean_txid}6304E8A2"
        )

