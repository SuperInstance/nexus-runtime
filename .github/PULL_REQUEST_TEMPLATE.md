## Description

Describe your changes clearly and concisely.

## Type

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / code cleanup
- [ ] Documentation update
- [ ] Hardware support (board config)

## Testing

Describe how you tested these changes:

- [ ] Unit tests pass (`python -m pytest --tb=short -q`)
- [ ] Firmware tests pass (`cd tests/firmware/build && cmake .. && make && ./test_firmware`)
- [ ] Linting passes (`ruff check nexus/ jetson/`)
- [ ] Type checks pass (`mypy nexus/ jetson/`)

## Checklist

- [ ] All tests pass with no regressions
- [ ] Documentation has been updated (if applicable)
- [ ] CHANGELOG.md has been updated (if applicable)
- [ ] No breaking changes to existing APIs or wire protocol
- [ ] Code follows project style (ruff, mypy)
- [ ] Security implications have been considered (if applicable)
