import os
import re

class FileServices:
    def saveFile(response, save_directory):
        content_disposition = response.headers.get('Content-Disposition', '')
        filename = re.findall('filename=(.+)', content_disposition)
        if not filename:
            raise ValueError('No filename found in response headers')
        file_path = os.path.join(save_directory, filename[0].strip('"'))
        with open(file_path, 'wb') as file:
            file.write(response.getvalue())

        return True