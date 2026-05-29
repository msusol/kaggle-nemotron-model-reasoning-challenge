# Citation format and registry

All citations across plans, README, and other project markdown must use the
`[cite:N]` inline tag and be registered in `docs/plans/CITATIONS.md`.

## Rules

1. **Never use bare URLs inline.** Always replace with `[cite:N]`.

2. **Reuse existing numbers.** Before assigning a new number, check
   `docs/plans/CITATIONS.md` for the URL. If it is already listed, use its
   existing `[cite:N]`.

3. **Sequential numbering for new citations.** New citations get the next
   integer after the current highest `[cite:N]` number in `docs/plans/CITATIONS.md`.
   Always derive this by scanning the file — never assume a fixed value.

4. **Register every new citation immediately.** After adding a `[cite:N]`
   inline, add the corresponding entry to `docs/plans/CITATIONS.md` in the same
   edit, in ascending numeric order:
   ```
   - `[cite:N]` Title of page — <https://url>
   ```

5. **Keep `docs/plans/CITATIONS.md` sorted by number.** Entries must appear in
   ascending order so gaps are visible and duplicates are easy to spot.

6. **One URL = one number.** Never assign two different numbers to the same
   URL. If the same source is cited in multiple files, the same `[cite:N]`
   is used everywhere.
