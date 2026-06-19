# Conspectum

Веб-сервис, который превращает аудиозапись лекции в транскрипт, структурированный LaTeX-конспект и PDF.

Conspectum локально распознаёт речь через `faster-whisper`, формирует академический конспект с помощью OpenAI-совместимого API, исправляет типовые ошибки сгенерированного LaTeX и предоставляет готовые материалы через веб-интерфейс.

## Возможности

- загрузка аудиофайла или прямой публичной ссылки на аудио;
- поддержка WAV, MP3, M4A, OGG, Opus, FLAC, AAC, MP4/M4B и WebM;
- локальная транскрибация через `faster-whisper`;
- автоматический выбор CUDA/CPU с безопасными fallback-сценариями;
- русский и английский язык, включая автоопределение;
- три уровня детализации: краткий, стандартный и подробный;
- обработка длинных лекций по чанкам с ограниченной параллельностью;
- основная и резервные LLM-модели;
- генерация заголовка, аннотации и структурированных разделов;
- восстановление Unicode-формул, таблиц, списков и LaTeX-окружений;
- сборка PDF через XeLaTeX, LuaLaTeX или pdfLaTeX;
- читаемый fallback-PDF через ReportLab при ошибке LaTeX-сборки;
- скачивание TXT, TEX, PDF или общего ZIP с метаданными;
- живой прогресс, журнал этапов и восстановление активной задачи после перезагрузки страницы;
- ограничения загрузки, SSRF-защита, rate limiting и безопасная выдача результатов.

## Как устроена обработка

```text
аудиофайл или URL
        │
        ▼
валидация и безопасная загрузка
        │
        ▼
локальный faster-whisper ──► транскрипт TXT
        │
        ▼
заголовок и аннотация через LLM
        │
        ▼
разбиение транскрипта и генерация разделов
        │
        ▼
локальная нормализация и восстановление LaTeX
        │
        ├──► TEX
        └──► LaTeX engine / ReportLab fallback ──► PDF
```

## Требования

- Python 3.11 или новее;
- FFmpeg для декодирования большинства аудиоформатов и предобработки;
- API-ключ OpenAI-совместимого провайдера;
- необязательно: NVIDIA GPU с рабочим CUDA runtime;
- необязательно: `xelatex`, `lualatex` или `pdflatex` для полноценной PDF-сборки.

Без GPU сервис работает на CPU, но распознавание моделью `large-v3` может быть значительно медленнее. Без LaTeX-движка пользователь всё равно получает транскрипт и TEX.

## Быстрый старт

### Windows PowerShell

```powershell
git clone <URL-ВАШЕГО-РЕПОЗИТОРИЯ>
cd Conspectum
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Откройте `.env`, замените `AI_API_KEY=replace-me`, затем запустите:

```powershell
.\.venv\Scripts\python.exe src\web.py
```

### Linux/macOS

```bash
git clone <URL-ВАШЕГО-РЕПОЗИТОРИЯ>
cd Conspectum
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python src/web.py
```

После запуска откройте [http://127.0.0.1:8000](http://127.0.0.1:8000).

Swagger UI доступен по адресу `http://127.0.0.1:8000/docs`, если `ENABLE_API_DOCS=1`.

## Конфигурация

Реальные настройки хранятся в локальном `.env`. Этот файл исключён из Git и не должен публиковаться.

Минимально необходимо указать:

```dotenv
AI_BASE_URL=https://openrouter.ai/api/v1
AI_API_KEY=replace-me
MODEL_NAME=openai/gpt-oss-120b:free
```

Основные параметры:

| Переменная | Назначение |
|---|---|
| `AI_BASE_URL` | URL OpenAI-совместимого Chat Completions API |
| `AI_API_KEY` | секретный ключ провайдера |
| `MODEL_NAME` | основная модель генерации текста |
| `AI_MODEL_FALLBACKS` | резервные модели через запятую |
| `APP_ENV` | `development` или `production` |
| `MAX_UPLOAD_BYTES` | максимальный размер загружаемого файла |
| `ALLOWED_HOSTS` | разрешённые Host headers в production |
| `PRELOAD_WHISPER` | предзагрузка Whisper при старте |
| `WHISPER_MODEL_SIZE` | модель Whisper, например `large-v3` |
| `WHISPER_DEVICE` | `auto`, `cuda` или `cpu` |
| `WHISPER_COMPUTE_TYPE` | compute type CTranslate2 |
| `WHISPER_DEFAULT_LANGUAGE` | language hint для распознавания |
| `CHUNK_TARGET_CHARS` | целевой размер части транскрипта |
| `CHUNK_MAX_TOKENS` | лимит ответа модели на часть |
| `CHUNK_PROCESS_CONCURRENCY` | число параллельных LLM-запросов |
| `ENABLE_LLM_POSTPROCESS` | финальная LLM-редактура полного TEX |

Все доступные настройки и безопасные значения по умолчанию находятся в [.env.example](.env.example).

## Результаты

Во время работы сервис создаёт:

```text
logs/<uuid>/       диагностические материалы обработки
static/results/    файлы, доступные пользователю для скачивания
```

Итоговый ZIP задачи может содержать:

```text
transcript.txt
result.tex
result.pdf
summary.txt
metadata.json
```

Runtime-каталоги, аудио и сгенерированные результаты исключены из Git.

## API

| Метод и путь | Назначение |
|---|---|
| `GET /` | веб-интерфейс |
| `POST /upload` | загрузка локального аудиофайла |
| `POST /upload-url` | запуск по прямой публичной ссылке |
| `GET /status/{task_id}` | состояние и прогресс задачи |
| `GET /bundle/{task_id}` | ZIP всех результатов |
| `GET /static/{filename}` | разрешённый статический файл или результат |

Задачи и rate-limit buckets хранятся в памяти процесса. Текущая архитектура рассчитана прежде всего на локальный или однопроцессный запуск.

## Структура репозитория

```text
src/web.py                       FastAPI, endpoints, безопасность и задачи
src/web_template.html            HTML-интерфейс
static/web.css                   стили интерфейса
static/web.js                    клиентская логика и polling
src/conspectum/summary.py        Whisper и обращения к AI API
src/conspectum/process.py        чанки, LaTeX repair pipeline и PDF
src/conspectum/gpu.py            диагностика NVIDIA/CUDA
src/conspectum/logger.py         интерфейс прогресса и артефактов
src/conspectum/prompts/          промпты и LaTeX-шаблон
scripts/check_gpu.py             CLI-диагностика GPU
tests/                           автоматические тесты
.github/workflows/ci.yml         проверки GitHub Actions
```

## Разработка и тестирование

Установите зависимости разработчика:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Запустите проверки:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Диагностика GPU:

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu.py
```

Pull request автоматически проверяется GitHub Actions на поддерживаемых версиях Python.

## Приватность и безопасность

- Аудио транскрибируется локально.
- Текст транскрипта или его части отправляются настроенному AI-провайдеру.
- Логи могут содержать содержание лекции и не должны публиковаться.
- `.env` с реальным API-ключом не должен попадать в Git.
- Для публичного развёртывания используйте HTTPS, `APP_ENV=production`, `ALLOWED_HOSTS` и внешний reverse proxy.

## Лицензия

Проект распространяется по лицензии BSD 3-Clause. См. [LICENSE](LICENSE).
