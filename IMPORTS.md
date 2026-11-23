# Source imports

## Scripture JSON format

The scripture importer accepts one or more books. Number fields are optional and default to their array position; when supplied, they must be positive and unique within their parent. The importer accepts both the canonical names (`title`, `number`, `text`) and the Book of Mormon dataset names (`book`, `chapter`, `verse`). Verse text is stored exactly as decoded from JSON, without trimming or whitespace normalization. Book-level descriptive fields, chapter references, and verse references are retained in locators. If `source_locale` is absent from the file, the project's default is used; new projects default to `en`. Override it with `--source-locale` when needed.

```json
{
  "title": "Genesis",
  "source_locale": "he",
  "books": [
    {
      "number": 1,
      "title": "Genesis",
      "chapters": [
        {
          "number": 1,
          "verses": [
            {"number": 1, "text": "In the beginning..."},
            {"number": 2, "original_text": "And the earth..."}
          ]
        }
      ]
    }
  ]
}
```

`text` and `original_text` are accepted verse string keys. For the alternate schema, the book name can be `book`, chapter ordinal can be `chapter`, and verse ordinal can be `verse`. The top-level `book` object is also accepted as a one-book shorthand when the root has no `books` array. Each verse is stored in `verses` and also gets one aligned `chunks` row with a stable `book:chapter:verse` key. The raw JSON file text and SHA-256 are retained on `Document`; verse content and its UTF-8 hash are retained at verse granularity.

## CLI

Initialize the database and create a project first:

```sh
ai-translator db upgrade
ai-translator project create "My Project" --source-locale en
ai-translator import scripture scripture.json --project "My Project"
ai-translator import document novel.txt --project "My Project" --source-locale en --domain literary
```

The `document` command currently registers UTF-8 plain text. Paragraph chunks retain their original separators, and concatenating the chunk strings reproduces the original document exactly. New formats can implement the `DocumentParser` protocol and be registered with `ParserRegistry`; the service and CLI do not depend on a particular document parser.

Import services validate the complete parsed structure before inserting rows. They flush rows but do not commit; the caller owns the transaction and can roll back the whole import on failure.

## Removing imported documents (testing)

These permanent cleanup commands require confirmation unless `--yes` is supplied. They remove document source units and all dependent translations, evaluations, reviews, experiment runs, and published outputs. Projects, glossaries, and experiments are retained. A failed foreign-key check rolls the transaction back.

```sh
ai-translator document list
ai-translator document remove DOCUMENT_UUID
ai-translator document clear
```

Pass `--yes` to either removal command for non-interactive use. For example:

```sh
ai-translator document remove DOCUMENT_UUID --yes
```
