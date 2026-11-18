import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class DeteksiApel:

    def __init__(self, root):
        self.root = root
        self.root.title("Deteksi Pembusukan Buah Apel")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f2f5")

        # Judul aplikasi
        judul = tk.Label(
            root,
            text="DETEKSI PEMBUSUKAN BUAH APEL",
            font=("Arial", 20, "bold"),
            bg="#f0f2f5"
        )
        judul.pack(pady=20)

        # Tombol pilih gambar
        tombol = tk.Button(
            root,
            text="Pilih Gambar Apel",
            command=self.pilih_gambar,
            font=("Arial", 12, "bold"),
            bg="#2878d0",
            fg="white",
            padx=20,
            pady=10
        )
        tombol.pack(pady=10)

        # Frame gambar
        frame = tk.Frame(root, bg="#f0f2f5")
        frame.pack(pady=20)

        frame_asli = tk.Frame(frame, bg="white")
        frame_asli.grid(row=0, column=0, padx=15)

        frame_hasil = tk.Frame(frame, bg="white")
        frame_hasil.grid(row=0, column=1, padx=15)

        # Gambar asli
        tk.Label(
            frame_asli,
            text="Gambar Asli",
            font=("Arial", 13, "bold"),
            bg="white"
        ).pack(pady=10)

        self.label_asli = tk.Label(
            frame_asli,
            text="Belum ada gambar",
            bg="white",
            width=55,
            height=25
        )
        self.label_asli.pack(padx=10, pady=10)

        # Hasil deteksi
        tk.Label(
            frame_hasil,
            text="Hasil Deteksi",
            font=("Arial", 13, "bold"),
            bg="white"
        ).pack(pady=10)

        self.label_hasil = tk.Label(
            frame_hasil,
            text="Belum ada hasil",
            bg="white",
            width=55,
            height=25
        )
        self.label_hasil.pack(padx=10, pady=10)

        # Status hasil
        self.label_status = tk.Label(
            root,
            text="Silakan pilih gambar apel.",
            font=("Arial", 14, "bold"),
            bg="#f0f2f5",
            fg="#333333"
        )
        self.label_status.pack(pady=20)

    # Fungsi menampilkan gambar
    def tampilkan_gambar(self, gambar, label):

        rgb = cv2.cvtColor(
            gambar,
            cv2.COLOR_BGR2RGB
        )

        tinggi, lebar = rgb.shape[:2]

        skala = min(
    320 / lebar,
    320 / tinggi
)

        ukuran = (
            int(lebar * skala),
            int(tinggi * skala)
        )

        rgb = cv2.resize(
            rgb,
            ukuran
        )

        foto = Image.fromarray(rgb)
        foto_tk = ImageTk.PhotoImage(foto)

        label.config(
    image=foto_tk,
    text="",
    width=320,
    height=320
)

        label.image = foto_tk

    # Fungsi memilih gambar
    def pilih_gambar(self):

        lokasi = filedialog.askopenfilename(
            title="Pilih Gambar Apel",
            filetypes=[
                ("File gambar", "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if not lokasi:
            return

        gambar = cv2.imread(lokasi)

        if gambar is None:
            messagebox.showerror(
                "Error",
                "Gambar tidak dapat dibaca."
            )
            return

        # Operasi geometri: mengubah ukuran gambar
        tinggi, lebar = gambar.shape[:2]

        skala = min(
            1,
            1000 / max(tinggi, lebar)
        )

        if skala < 1:
            gambar = cv2.resize(
                gambar,
                (
                    int(lebar * skala),
                    int(tinggi * skala)
                )
            )

        self.proses_gambar(gambar)

    # Fungsi pengolahan citra
    def proses_gambar(self, gambar):

        # 1. Menampilkan gambar asli
        self.tampilkan_gambar(
            gambar,
            self.label_asli
        )

        # 2. Filtering untuk mengurangi noise
        gambar_filter = cv2.medianBlur(
            gambar,
            5
        )

        # 3. Konversi warna BGR ke HSV
        hsv = cv2.cvtColor(
            gambar_filter,
            cv2.COLOR_BGR2HSV
        )

        # 4. Segmentasi objek apel
        batas_objek_bawah = np.array([0, 35, 20])
        batas_objek_atas = np.array([179, 255, 245])

        mask_objek = cv2.inRange(
            hsv,
            batas_objek_bawah,
            batas_objek_atas
        )

        # 5. Membersihkan mask objek
        kernel_objek = np.ones(
            (7, 7),
            np.uint8
        )

        mask_objek = cv2.morphologyEx(
            mask_objek,
            cv2.MORPH_CLOSE,
            kernel_objek
        )

        # 6. Mengambil objek terbesar
        kontur_objek, _ = cv2.findContours(
            mask_objek,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        hasil = gambar.copy()

        if len(kontur_objek) == 0:

            self.label_status.config(
                text="APEL TIDAK TERDETEKSI",
                fg="#c0392b"
            )

            self.tampilkan_gambar(
                hasil,
                self.label_hasil
            )

            return

        kontur_apel = max(
            kontur_objek,
            key=cv2.contourArea
        )

        # Membuat mask khusus area apel
        mask_apel = np.zeros(
            gambar.shape[:2],
            dtype=np.uint8
        )

        cv2.drawContours(
            mask_apel,
            [kontur_apel],
            -1,
            255,
            cv2.FILLED
        )

        # 7. Mengurangi area tepi apel
        kernel_tepi = np.ones(
            (21, 21),
            np.uint8
        )

        mask_apel_dalam = cv2.erode(
            mask_apel,
            kernel_tepi,
            iterations=1
        )

        # 8. Segmentasi warna cokelat
        batas_cokelat_bawah = np.array([8, 75, 30])
        batas_cokelat_atas = np.array([28, 255, 175])

        mask_cokelat = cv2.inRange(
            hsv,
            batas_cokelat_bawah,
            batas_cokelat_atas
        )

        # Membatasi deteksi di dalam permukaan apel
        mask_bercak = cv2.bitwise_and(
            mask_cokelat,
            mask_apel_dalam
        )

        # 9. Menghilangkan noise kecil
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask_bercak = cv2.morphologyEx(
            mask_bercak,
            cv2.MORPH_OPEN,
            kernel
        )

        # 10. Memperbaiki area bercak
        mask_bercak = cv2.morphologyEx(
            mask_bercak,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 11. Mendeteksi kontur bercak
        kontur_bercak, _ = cv2.findContours(
            mask_bercak,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        ada_bercak = False

        luas_apel = cv2.contourArea(kontur_apel)

        luas_minimum = max(
            100,
            int(luas_apel * 0.001)
        )

        for area in kontur_bercak:

            luas = cv2.contourArea(area)

            if luas > luas_minimum:

                ada_bercak = True

                x, y, w, h = cv2.boundingRect(area)

                # Menandai area bercak
                cv2.rectangle(
                    hasil,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    hasil,
                    "Bercak",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # 12. Menentukan hasil deteksi
        if ada_bercak:

            status = "TERINDIKASI BUSUK"
            warna = "#c0392b"

        else:

            status = "TIDAK TERDETEKSI BUSUK"
            warna = "#238443"

        self.label_status.config(
            text="Hasil Deteksi: " + status,
            fg=warna
        )

        # 13. Menampilkan hasil
        self.tampilkan_gambar(
            hasil,
            self.label_hasil
        )


# Menjalankan aplikasi
if __name__ == "__main__":

    root = tk.Tk()

    aplikasi = DeteksiApel(root)

    root.mainloop()