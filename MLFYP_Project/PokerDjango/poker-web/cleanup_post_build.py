def create_paths():
    ROOT_DIR = '/home/gary/Desktop/Dev/Python/MLFYP_Project/PokerDjango'
    path_from = os.path.join(ROOT_DIR, 'poker-web', 'build', 'static', '')
    path_to = os.path.join(ROOT_DIR, 'static', '')
    templates_dir = os.path.join(ROOT_DIR, 'poker', 'templates', 'react', '')
    return path_from, path_to, templates_dir

def move_content(path_from, path_to):
    files = os.listdir(path_from)
    for f in files:
        destined_path = os.path.join(path_to, f)
        try:
            os.remove(destined_path)
        except IsADirectoryError:
            shutil.rmtree(destined_path)

        shutil.move(os.path.join(path_from, f), path_to)

def rename_static_references(static_dir, templates_dir):
    files_js = os.listdir(os.path.join(static_dir, 'js'))
    files_css = os.listdir(os.path.join(static_dir, 'css'))
    static_file_names = {
        "js_plain": list(filter(re.compile(r'^[0-9]+.*\.js$').match, files_js))[0],
        "js_main": list(filter(re.compile(r'^(main)+.*\.js$').match, files_js))[0],
        "css": list(filter(re.compile("^main.*css$").match, files_css))[0]
    }

    js_file_text=open(os.path.join(templates_dir, 'js.html'), 'r').read()
    js_file_write= open(os.path.join(templates_dir, 'js.html'), 'w')
    js_file_text = re.sub(r'(js/[0-9]).*(js)', "js/"+static_file_names["js_plain"], js_file_text)
    js_file_text = re.sub(r'(js/main).*(js)', "js/"+static_file_names["js_main"], js_file_text)
    js_file_write.write(js_file_text)
    js_file_write.close()

    css_file=open(os.path.join(templates_dir, 'css.html'), 'r').read()
    css_file_write= open(os.path.join(templates_dir, 'css.html'), 'w')
    css_file = re.sub(r'(<link href="/static/css/main.[0-9a-zA-Z]+.chunk.css" rel="stylesheet">)', '<link href="/static/css/{}" rel="stylesheet">'.format(static_file_names["css"]), css_file)
    css_file_write.write(css_file)
    css_file_write.close()


import os, shutil, re, sys
path_from, path_to, templates_dir = create_paths()
move_content(path_from, path_to)
rename_static_references(path_to, templates_dir)