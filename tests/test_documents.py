"""Structural checks for the authored Office documents (no Office required)."""

from pathlib import Path
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
DOCUMENTS = ('my_research_work', 'my_research_work_def', 'Research_work',
             'Technical_specifications', 'Text_for_presentation')


def paragraphs(root):
    # A text box's containing paragraph repeats its children's text otherwise.
    return [p for p in root.iter('{' + W + '}p')
            if not p.findall('.//w:p', NS)]


def text(paragraph):
    return ''.join(node.text or '' for node in paragraph.iter('{' + W + '}t'))


class DocumentTests(unittest.TestCase):
    def test_archives_and_xml_are_valid(self):
        for name in DOCUMENTS:
            with self.subTest(document=name), ZipFile(ROOT / 'docs' / (name + '.docx')) as archive:
                self.assertIsNone(archive.testzip())
                for entry in archive.namelist():
                    if entry.endswith(('.xml', '.rels')):
                        ET.fromstring(archive.read(entry))

    def test_fields_refresh_when_opened(self):
        for name in DOCUMENTS:
            with self.subTest(document=name), ZipFile(ROOT / 'docs' / (name + '.docx')) as archive:
                settings = ET.fromstring(archive.read('word/settings.xml'))
                field = settings.find('w:updateFields', NS)
                self.assertIsNotNone(field)
                self.assertEqual(field.get('{' + W + '}val'), 'true')

    def test_cover_header_has_separate_lines(self):
        headers = 0
        for name in DOCUMENTS:
            with ZipFile(ROOT / 'docs' / (name + '.docx')) as archive:
                root = ET.fromstring(archive.read('word/document.xml'))
            for paragraph in paragraphs(root):
                value = text(paragraph)
                if value.startswith('\u041c\u0438\u043d\u0438\u0441\u0442\u0435\u0440\u0441\u0442\u0432\u043e') and '\u0424\u0410\u041a\u0423\u041b\u042c\u0422\u0415\u0422' in value:
                    headers += 1
                    with self.subTest(document=name):
                        self.assertEqual(len(paragraph.findall('.//w:br', NS)), 7)
                        self.assertEqual(paragraph.find('w:pPr/w:jc', NS).get('{' + W + '}val'), 'center')
                        self.assertEqual(paragraph.find('w:r/w:rPr/w:sz', NS).get('{' + W + '}val'), '22')
        self.assertGreater(headers, 0)

    def test_code_is_not_one_oversized_paragraph(self):
        for name in DOCUMENTS:
            with ZipFile(ROOT / 'docs' / (name + '.docx')) as archive:
                root = ET.fromstring(archive.read('word/document.xml'))
            for paragraph in paragraphs(root):
                if text(paragraph).startswith(('def ', 'from pathlib import Path', 'import nltk')):
                    with self.subTest(document=name, code=text(paragraph)[:60]):
                        self.assertLessEqual(len(paragraph.findall('.//w:br', NS)), 10)


if __name__ == '__main__':
    unittest.main()
