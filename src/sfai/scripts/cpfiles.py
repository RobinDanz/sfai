from pathlib import Path
import os
from typing import List
import shutil

def copy(source : Path, destination : Path, extensions: List[str] = [], move : bool = False):
    """Copies source subfiles into destination

    Args:
        source (Path): Source folder
        destination (Path): Destination folder. Created if not exists
        extensions (List[str], optional): Filter files on extensions. Defaults to [].
        move (bool, optional): If true, moves the file instead of copy. Defaults to False.

    Raises:
        NotADirectoryError: Raised if source does not exist or is not a folder.
    """
    
    files_to_move = []
    extensions = [f".{ext.lstrip('.')}" for ext in extensions]

    if not source.exists() or not source.is_dir():
        raise NotADirectoryError('Source folder not found. Make sure that it exists.')

    for dirpath, dirnames, filenames in os.walk(source):
        for filename in filenames:
            _, ext = os.path.splitext(filename)

            if not extensions or ext.lower() in extensions:
                filepath = os.path.join(dirpath, filename)
                
                files_to_move.append(Path(filepath))

    destination.mkdir(parents=True, exist_ok=True)

    if os.listdir(destination):
        print(f'{destination.as_posix()} contain files. Moving {len(files_to_move)} ? y/n')

        response = input()

        allowed = ['y', 'n']

        while str.lower(response.strip()) not in allowed:
            print('Please type y or n.')
            response = input()

        if response == 'n':
            print('Aborting.')
            exit(0)

    action = shutil.move if move else shutil.copy

    for file in files_to_move:
        action(file, destination)
        
    print(f'Moved {len(files_to_move)} files to {destination}')
    
    
    