# Documentation

### Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally. To install, use the following command

```
npm i -g mintlify
```

Run the following command at the root of your documentation (where mint.json is)

```
mintlify dev
```

#### Troubleshooting

- **Mintlify dev isn't running**: Re-install the CLI using `npm i -g mintlify@latest`
- **Page loads as a 404**: Make sure you are running in a folder with `mint.json`
