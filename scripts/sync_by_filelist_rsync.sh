#!/usr/bin/env bash
# 按“文件/目录清单”批量同步到多台目标机（rsync / tar 传输）
#
# 默认配置对齐文档《三台主机按清单rsync同步指南.md》118-128 行：
#   - base: /home/tonera/ai/models
#   - list: /home/tonera/ai/models/sync_files_image.txt
#   - targets:
#       tonera@192.168.0.112:/home/tonera/models/
#       tonera@192.168.0.107:/home/tonera/models/
#       tonera@192.168.0.102:/home/tonera/aimodels/models
#
# 用法示例：
#   bash scripts/sync_by_filelist_rsync.sh --dry-run
#   bash scripts/sync_by_filelist_rsync.sh --base /home/tonera/ai/models --list /home/tonera/ai/models/sync_files_image.txt \
#     --targets tonera@192.168.0.112:/home/tonera/models/,tonera@192.168.0.107:/home/tonera/models/,tonera@192.168.0.102:/home/tonera/aimodels/models
#
# 提示：可执行权限
#   chmod +x scripts/sync_by_filelist_rsync.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
按文件/目录清单批量同步到多台目标机。

必选/常用参数：
  --base <DIR>           基准目录（清单内路径相对该目录）
  --list <FILE>          清单文件（相对路径/目录建议写成相对 base 的路径）
  --targets <T1,T2,...>  目标列表，逗号分隔。每项形如：user@host:/abs/dest/dir

可选参数：
  --sync-kind <models|loras|weights>
                         同步对象类型（默认 models）。
                         仅在未显式传 --targets 时生效，用于选择三台主机的目标目录。
  --expand-dirs          展开清单里的“目录项”（以 / 结尾）为递归文件清单再同步（默认开启）。
  --no-expand-dirs       禁用目录展开：目录项只会创建目录本身，不会同步其内部文件。
  --lint-list            同步前检查清单：列出每个目录项的实际文件数/被点名文件数，并提示高风险项。
  -n, --dry-run          预演（rsync 使用 -n；tar 模式仅打印将要执行的动作）
  --transport <rsync|tar> 传输方式（默认 rsync）
  --ssh-opts "<...>"     传给 ssh 的参数字符串，例如：--ssh-opts "-p 2222"
  --add-file <PATH>      单文件/目录增量模式：
                         1) 将 PATH（相对 --base 的路径）追加写入 --list 末尾（独立一行）
                         2) 执行 python inference/download/init_sha256_index.py \
                              --models-dir <base> --files-from <list> \
                              --merge --dedupe --skip-indexed --prune-missing
                         3) 仅把 <base>/models_sha256.json + PATH 传到所有 targets
                         注：此模式固定使用 rsync（忽略 --transport=tar）
  --tar-verbose          tar 模式下启用详细输出（tar -v）
  --show-skipped         tar 模式下预过滤清单，缺失项会打印 SKIP（rsync 模式仍交给 --ignore-missing-args）
  -h, --help             显示帮助

说明：
  - rsync 模式参数对齐文档：-avh --info=progress2 --size-only --ignore-missing-args --files-from=... --relative
  - 目录项递归同步说明：
      - 旧行为（rsync --files-from）：目录项只会创建目录，不保证递归同步目录内容。
      - 本脚本默认开启 --expand-dirs：会先把目录项展开为该目录下所有文件/链接，再喂给 rsync。
      - 如需“仅创建空目录”，使用 --no-expand-dirs。
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

BASE=""
LIST=""
TRANSPORT="rsync"
DRY_RUN=0
SSH_OPTS_STR=""
TAR_VERBOSE=0
SHOW_SKIPPED=0
ADD_FILE=""
SYNC_KIND="models"
EXPAND_DIRS=1
LINT_LIST=0

BASE_PROVIDED=0
LIST_PROVIDED=0

TARGETS=()

add_targets_from_arg() {
  local arg="${1:-}"
  local part
  IFS=',' read -r -a _parts <<<"$arg"
  for part in "${_parts[@]}"; do
    # 去掉前后空白
    part="${part#"${part%%[![:space:]]*}"}"
    part="${part%"${part##*[![:space:]]}"}"
    [[ -n "$part" ]] && TARGETS+=("$part")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync-kind|--sync-type|--kind)
      SYNC_KIND="${2:-}"; shift 2
      ;;
    --base)
      BASE="${2:-}"; shift 2
      BASE_PROVIDED=1
      ;;
    --list|--files-from)
      LIST="${2:-}"; shift 2
      LIST_PROVIDED=1
      ;;
    --targets)
      add_targets_from_arg "${2:-}"; shift 2
      ;;
    --transport)
      TRANSPORT="${2:-}"; shift 2
      ;;
    --expand-dirs)
      EXPAND_DIRS=1; shift
      ;;
    --no-expand-dirs|--no-expand-dir|--no-expand)
      EXPAND_DIRS=0; shift
      ;;
    --lint-list|--lint)
      LINT_LIST=1; shift
      ;;
    -n|--dry-run)
      DRY_RUN=1; shift
      ;;
    --ssh-opts)
      SSH_OPTS_STR="${2:-}"; shift 2
      ;;
    --add-file)
      ADD_FILE="${2:-}"; shift 2
      ;;
    --tar-verbose)
      TAR_VERBOSE=1; shift
      ;;
    --show-skipped)
      SHOW_SKIPPED=1; shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数：$1（用 --help 查看用法）"
      ;;
  esac
done

# 默认值（对齐文档 118-128）
[[ -n "$BASE" ]] || BASE="/home/tonera/ai/models"
[[ -n "$LIST" ]] || LIST="/home/tonera/ai/models/sync_files_image.txt"
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  # 目标机配置表：后续新增机器，直接在这里追加一行即可
  case "$SYNC_KIND" in
    models|"")
      TARGETS+=("tonera@192.168.0.112:/home/tonera/models/")
      TARGETS+=("tonera@192.168.0.107:/home/tonera/models/")
      TARGETS+=("tonera@192.168.0.102:/home/tonera/aimodels/models")
      ;;
    loras)
      TARGETS+=("tonera@192.168.0.112:/home/tonera/loras/")
      TARGETS+=("tonera@192.168.0.107:/home/tonera/loras/")
      TARGETS+=("tonera@192.168.0.102:/home/tonera/project/aiservice/diffusers/loras")
      ;;
    weights)
      TARGETS+=("tonera@192.168.0.112:/home/tonera/weights/")
      TARGETS+=("tonera@192.168.0.107:/home/tonera/weights/")
      TARGETS+=("tonera@192.168.0.102:/home/tonera/project/aiservice/diffusers/weights")
      ;;
    *)
      die "--sync-kind 仅支持 models/loras/weights，当前：$SYNC_KIND"
      ;;
  esac
fi

if [[ "$SYNC_KIND" != "models" && $BASE_PROVIDED -eq 0 ]]; then
  echo "WARN: --sync-kind=$SYNC_KIND 但未显式传 --base，当前使用默认 BASE=$BASE（请确认是否正确）" >&2
fi
if [[ "$SYNC_KIND" != "models" && $LIST_PROVIDED -eq 0 ]]; then
  echo "WARN: --sync-kind=$SYNC_KIND 但未显式传 --list，当前使用默认 LIST=$LIST（请确认是否正确）" >&2
fi

command -v rsync >/dev/null 2>&1 || die "找不到 rsync，请先安装/确认 PATH"
[[ -d "$BASE" ]] || die "BASE 不存在或不是目录：$BASE"
[[ -f "$LIST" ]] || die "清单文件不存在：$LIST"

BASE="${BASE%/}"

RSYNC_PROGRESS_OPT=()
RSYNC_IGNORE_MISSING_OPT=()
detect_rsync_progress_opt() {
  # rsync 2.6.x（macOS 常见）不支持 --info=progress2
  if rsync --help 2>&1 | grep -q 'progress2'; then
    RSYNC_PROGRESS_OPT=(--info=progress2)
  else
    RSYNC_PROGRESS_OPT=(--progress)
  fi
}
detect_rsync_ignore_missing_opt() {
  if rsync --help 2>&1 | grep -q 'ignore-missing-args'; then
    RSYNC_IGNORE_MISSING_OPT=(--ignore-missing-args)
  else
    RSYNC_IGNORE_MISSING_OPT=()
  fi
}
detect_rsync_progress_opt
detect_rsync_ignore_missing_opt

SSH_OPTS=()
if [[ -n "$SSH_OPTS_STR" ]]; then
  # 按空白切分（适配常见写法：--ssh-opts "-p 2222 -i ~/.ssh/id_ed25519"）
  # shellcheck disable=SC2206
  SSH_OPTS=($SSH_OPTS_STR)
fi

parse_target_host_dest() {
  local target="$1"
  local host dest
  [[ "$target" == *:* ]] || die "target 必须形如 user@host:/abs/dest/dir ：$target"
  host="${target%%:*}"
  dest="${target#*:}"
  dest="${dest%/}"
  [[ -n "$host" && -n "$dest" ]] || die "target 解析失败：$target"
  printf '%s\t%s\n' "$host" "$dest"
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  return 1
}

normalize_add_file_to_rel() {
  # 输出相对 BASE 的路径（允许文件或目录；如果用户传了末尾 / 则保留）
  local in="$1"
  local rel abs

  [[ -n "$in" ]] || die "--add-file 不能为空"

  # ~ 扩展（仅支持 ~/...）
  if [[ "$in" == "~/"* ]]; then
    in="$HOME/${in#~/}"
  fi

  # 禁止 ../ 越界
  if [[ "$in" == ".." || "$in" == "../"* || "$in" == *"/../"* || "$in" == *"/.." ]]; then
    die "--add-file 不允许包含 '..' 路径段：$in"
  fi

  if [[ "$in" == /* ]]; then
    abs="$in"
    case "$abs" in
      "$BASE"/*) rel="${abs#"$BASE"/}" ;;
      "$BASE") die "--add-file 不能等于 BASE 自身：$in" ;;
      *) die "--add-file 必须位于 BASE 目录下（或传相对路径）。BASE=$BASE, add-file=$in" ;;
    esac
  else
    rel="$in"
  fi

  # 去掉开头的 ./（可能重复）
  while [[ "$rel" == ./* ]]; do
    rel="${rel#./}"
  done
  [[ -n "$rel" ]] || die "--add-file 解析后为空：$in"

  printf '%s\n' "$rel"
}

sanitize_list_line() {
  # 1) 去掉 CRLF
  # 2) 忽略空行与注释行
  # 3) 去掉开头 ./（可能重复）
  # 4) 禁止包含 .. 越界
  local p="$1"
  p="${p%$'\r'}"
  [[ -z "$p" ]] && return 1
  [[ "$p" =~ ^[[:space:]]*# ]] && return 1
  # 去掉前后空白（只处理常见情况）
  p="${p#"${p%%[![:space:]]*}"}"
  p="${p%"${p##*[![:space:]]}"}"
  while [[ "$p" == ./* ]]; do
    p="${p#./}"
  done
  if [[ "$p" == ".." || "$p" == "../"* || "$p" == *"/../"* || "$p" == *"/.." ]]; then
    die "清单中不允许包含 '..' 路径段：$p"
  fi
  printf '%s\n' "$p"
  return 0
}

lint_list_dirs() {
  local list_file="$1"
  local p dir actual listed

  echo "==> [lint-list] base=$BASE list=$list_file"
  echo "==> [lint-list] format: <dir/>  actual=<N>  listed=<M>  (listed=清单里点名的子文件/链接条数)"
  (
    cd "$BASE" || exit 1
    while IFS= read -r p || [[ -n "$p" ]]; do
      p="$(sanitize_list_line "$p" || true)"
      [[ -z "$p" ]] && continue

      [[ "$p" == */ ]] || continue
      dir="${p%/}"
      [[ -d "$dir" ]] || continue

      actual="$(find "$dir" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d '[:space:]')"
      # listed: 清单中以 "dir/" 开头、且不是目录项本身（不以 / 结尾）的条目数
      listed="$(grep -E "^${dir}/" "$list_file" 2>/dev/null | grep -v '/$' | wc -l | tr -d '[:space:]')"

      printf '%s/  actual=%s  listed=%s\n' "$dir" "$actual" "$listed"
      if [[ "$actual" != "0" && "$listed" == "0" && $EXPAND_DIRS -eq 0 ]]; then
        echo "WARN: 目录项 '$dir/' 下存在文件，但当前 --no-expand-dirs，会导致该目录内容不会被同步。" >&2
      fi
    done
  ) | sort
}

build_expanded_files_from_list() {
  # 将 LIST 展开成 rsync --files-from 可用的“文件/链接 + 必要目录项”清单。
  # - 保留原有文件条目
  # - 对目录条目（以 / 结尾）：先输出目录本身（保证空目录可创建），再递归输出其下所有文件/链接
  local list_file="$1"
  local out_file="$2"
  local p dir

  (
    cd "$BASE" || exit 1
    while IFS= read -r p || [[ -n "$p" ]]; do
      p="$(sanitize_list_line "$p" || true)"
      [[ -z "$p" ]] && continue

      if [[ "$p" == */ ]]; then
        dir="${p%/}"
        [[ -d "$dir" ]] || continue
        # 先保留目录项本身（确保空目录同步时也能创建）
        printf '%s/\n' "$dir"
        # 再递归输出目录下的文件/链接
        find "$dir" \( -type f -o -type l \) -print
      else
        printf '%s\n' "$p"
      fi
    done <"$list_file"
  ) | sort -u >"$out_file"
}

add_file_incremental_mode() {
  local rel rel_for_sync abs_path index_file py

  rel="$(normalize_add_file_to_rel "$ADD_FILE")"
  rel_for_sync="${rel%/}"
  abs_path="$BASE/$rel_for_sync"

  [[ -n "$rel_for_sync" ]] || die "--add-file 非法（解析后为空）: $ADD_FILE"
  if [[ ! -e "$abs_path" ]]; then
    die "--add-file 指定的路径在 BASE 下不存在：$abs_path"
  fi

  echo "==> [add-file] append to list: $LIST  <<  $rel"
  printf '%s\n' "$rel" >>"$LIST"

  py="$(find_python)" || die "找不到 python3/python，无法运行 init_sha256_index.py"
  echo "==> [add-file] update sha256 index: $BASE/models_sha256.json"
  "$py" "$REPO_ROOT/inference/download/init_sha256_index.py" \
    --models-dir "$BASE" \
    --files-from "$LIST" \
    --merge --dedupe --skip-indexed --prune-missing

  index_file="$BASE/models_sha256.json"
  [[ -f "$index_file" ]] || die "未生成索引文件：$index_file"

  # 增量同步：只同步 models_sha256.json + add-file（文件/目录）
  local -a rsync_common
  rsync_common=(
    -avh
    ${RSYNC_PROGRESS_OPT[@]+"${RSYNC_PROGRESS_OPT[@]}"}
    ${RSYNC_IGNORE_MISSING_OPT[@]+"${RSYNC_IGNORE_MISSING_OPT[@]}"}
    --relative
  )
  if (( DRY_RUN )); then
    rsync_common+=(-n)
  fi
  if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
    rsync_common+=(-e "ssh ${SSH_OPTS[*]}")
  fi

  for t in "${TARGETS[@]}"; do
    local host dest
    IFS=$'\t' read -r host dest <<<"$(parse_target_host_dest "$t")"

    echo "==> [add-file] ensure remote dir: $host:$dest"
    if ! (( DRY_RUN )); then
      ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$dest'"
    fi

    echo "==> [add-file][rsync] $BASE/./models_sha256.json + $BASE/./$rel_for_sync  ->  $t"
    rsync "${rsync_common[@]}" \
      "$BASE/./models_sha256.json" \
      "$BASE/./$rel_for_sync" \
      "$t"
  done
}

# 单文件增量模式：按用户要求固定流程后退出
if [[ -n "$ADD_FILE" ]]; then
  TRANSPORT="rsync"
  add_file_incremental_mode
  exit 0
fi

sync_one_rsync() {
  local target="$1"
  local -a rsync_common

  rsync_common=(
    -avh
    ${RSYNC_PROGRESS_OPT[@]+"${RSYNC_PROGRESS_OPT[@]}"}
    --size-only
    ${RSYNC_IGNORE_MISSING_OPT[@]+"${RSYNC_IGNORE_MISSING_OPT[@]}"}
    "--files-from=$LIST"
    --relative
  )

  if (( DRY_RUN )); then
    rsync_common+=(-n)
  fi

  if [[ ${#SSH_OPTS[@]} -gt 0 ]]; then
    rsync_common+=(-e "ssh ${SSH_OPTS[*]}")
  fi

  echo "==> [rsync] $BASE/  ->  $target"
  rsync "${rsync_common[@]}" "$BASE/" "$target"
}

sync_one_tar() {
  local target="$1"
  local host dest
  local list_to_use="$LIST"
  local -a tar_create

  [[ "$target" == *:* ]] || die "tar 模式下 target 必须形如 user@host:/abs/dest/dir ：$target"
  host="${target%%:*}"
  dest="${target#*:}"
  dest="${dest%/}"
  [[ -n "$host" && -n "$dest" ]] || die "target 解析失败：$target"

  if (( SHOW_SKIPPED )); then
    local tmp
    tmp="$(mktemp)"
    # 在函数内设置 trap，避免多目标时提前清理；函数退出时清理当前 tmp
    trap 'rm -f "$tmp"' RETURN

    while IFS= read -r p || [[ -n "$p" ]]; do
      [[ -z "$p" ]] && continue
      [[ "$p" =~ ^[[:space:]]*# ]] && continue
      # 去掉 CRLF
      p="${p%$'\r'}"
      if [[ -e "$BASE/$p" ]]; then
        printf '%s\n' "$p" >>"$tmp"
      else
        printf 'SKIP (missing): %s\n' "$p" >&2
      fi
    done <"$LIST"
    list_to_use="$tmp"
  fi

  tar_create=(tar -C "$BASE" -T "$list_to_use" -cf -)
  if (( TAR_VERBOSE )); then
    tar_create=(tar -C "$BASE" -T "$list_to_use" -cvf -)
  fi

  if (( DRY_RUN )); then
    echo "==> [tar][dry-run] $BASE (list=$LIST)  ->  $host:$dest"
    echo "    （tar 模式 dry-run 仅打印动作，不实际传输）"
    return 0
  fi

  echo "==> [tar] $BASE (list=$LIST)  ->  $host:$dest"
  if command -v pv >/dev/null 2>&1; then
    "${tar_create[@]}" | pv -ptearb | ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$dest' && tar -C '$dest' -xpf -"
  else
    "${tar_create[@]}" | ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$dest' && tar -C '$dest' -xpf -"
  fi
}

case "$TRANSPORT" in
  rsync)
    # rsync + --files-from: 目录项默认不会递归同步内容，这里提供自动展开能力
    if (( LINT_LIST )); then
      lint_list_dirs "$LIST"
    fi
    if (( EXPAND_DIRS )); then
      tmp_list="$(mktemp)"
      trap 'rm -f "$tmp_list"' RETURN
      echo "==> [expand-dirs] building expanded list -> $tmp_list"
      build_expanded_files_from_list "$LIST" "$tmp_list"
      echo "==> [expand-dirs] expanded list lines: $(wc -l <"$tmp_list" | tr -d '[:space:]')"
      LIST="$tmp_list"
    fi
    for t in "${TARGETS[@]}"; do
      sync_one_rsync "$t"
    done
    ;;
  tar)
    for t in "${TARGETS[@]}"; do
      sync_one_tar "$t"
    done
    ;;
  *)
    die "不支持的 --transport：$TRANSPORT（仅支持 rsync / tar）"
    ;;
esac

