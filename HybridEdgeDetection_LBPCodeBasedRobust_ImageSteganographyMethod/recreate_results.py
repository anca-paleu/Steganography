import os
from config import COVER_DIR, N_BITS, MASSIVE_MESSAGE, LONG_MESSAGE
from histograms import plot_pdh, plot_intensity_histogram
from results import show_metrics_table, show_ttest_table
from results_restoration import show_restored_metrics_table_as_article, show_restored_metrics_table_corrected

def generate_figures():
    img_lena = os.path.join(COVER_DIR, 'lena_color.tiff')
    img_baboon = os.path.join(COVER_DIR, '4.2.03.tiff')
    img_wheel = os.path.join(COVER_DIR, '5.2.08.tiff')
    img_f16 = os.path.join(COVER_DIR, '4.2.05.tiff')

    if os.path.exists(img_lena):
        plot_pdh(img_lena, 'Lena', MASSIVE_MESSAGE, N_BITS)

    if os.path.exists(img_baboon):
        plot_pdh(img_baboon, 'Baboon', MASSIVE_MESSAGE, N_BITS)

    if os.path.exists(img_wheel):
        plot_intensity_histogram(img_wheel, 'Wheel', MASSIVE_MESSAGE, N_BITS)

    if os.path.exists(img_f16):
        plot_intensity_histogram(img_f16, 'F16', MASSIVE_MESSAGE, N_BITS)

def generate_tables():
    show_metrics_table(cover_dir=COVER_DIR, secret_text=LONG_MESSAGE, n_bits=N_BITS)
    show_ttest_table(cover_dir=COVER_DIR, secret_text=LONG_MESSAGE, n_bits=N_BITS)

    show_restored_metrics_table_as_article(cover_dir=COVER_DIR, secret_text=LONG_MESSAGE, n_bits=N_BITS)
    show_restored_metrics_table_corrected(cover_dir=COVER_DIR, secret_text=LONG_MESSAGE, n_bits=N_BITS)

if __name__ == "__main__":
    generate_figures()
    generate_tables()