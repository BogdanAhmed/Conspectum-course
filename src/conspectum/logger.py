import enum
import typing


class Logger:
    class FileType(enum.Enum):
        TEXT = 0
        TEX = 1
        PDF = 2
        AUDIO = 3

        def get_extension(self):
            match self:
                case Logger.FileType.TEXT:
                    return "txt"
                case Logger.FileType.TEX:
                    return "tex"
                case Logger.FileType.PDF:
                    return "pdf"
                case Logger.FileType.AUDIO:
                    return "wav"
                case _:
                    assert False, f"Unsupported file type: {self}"

        def writing_mode(self):
            match self:
                case Logger.FileType.TEXT | Logger.FileType.TEX:
                    return "w"
                case Logger.FileType.PDF | Logger.FileType.AUDIO:
                    return "wb"
                case _:
                    assert False, f"Unsupported file type: {self}"

    def __init__(self, out_folder: typing.Optional[str] = None):
        self.out_folder_ = out_folder

    async def file(self, key: str, output, type: "Logger.FileType"):
        if not self.out_folder_:
            return

        file_path = f"{self.out_folder_}/{key}.{type.get_extension()}"

        if type in (Logger.FileType.TEXT, Logger.FileType.TEX):
            with open(file_path, type.writing_mode(), encoding="utf-8", errors="replace") as file:
                file.write(output)
        else:
            with open(file_path, type.writing_mode()) as file:
                file.write(output)

    async def partial_result(self, text: str):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))

    async def stage(self, stage: str, progress: typing.Optional[int] = None):
        return

    async def progress(self, completed: int, total: int):
        if total <= 0:
            await self.partial_result("Progress: 0%")
            return
        await self.partial_result(f"Progress: {round(completed / total * 100)}%")
