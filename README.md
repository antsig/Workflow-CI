# Workflow CI & MLflow Project

Folder ini mendemonstrasikan implementasi **MLProject** untuk membungkus kode pelatihan (training) ke dalam sebuah format yang siap dijalankan ulang (reproducible) dan diintegrasikan dengan Continuous Integration (CI).

## Apa itu MLproject?

`MLproject` (di dalam folder `MLProject/MLproject`) adalah file konfigurasi dari MLflow yang mendefinisikan lingkungan eksekusi (Conda/Virtualenv), dependensi, dan berbagai _entry points_.

## Konfigurasi Entry Points

Menjalankan project ini akan secara otomatis memanggil command yang ditentukan, misalnya:

```yaml
entry_points:
  main:
    parameters:
      data_file: { type: string, default: "none" }
    command: "python modelling.py {data_file}"
```

Hal ini memungkinkan otomasi eksekusi model pada tahap _Continuous Integration (CI)_ melalui runner di GitHub Actions, sehingga setiap perubahan pada kode model atau _hyperparameter_ dapat ditraining lalu divalidasi dengan konsisten.
