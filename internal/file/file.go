//
// Copyright (C) 2019-2022 vdaas.org vald team <vald@vdaas.org>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Package file provides file I/O functionality
package file

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/kpango/fastime"
	"github.com/vdaas/vald/internal/errgroup"
	"github.com/vdaas/vald/internal/errors"
	"github.com/vdaas/vald/internal/io"
	"github.com/vdaas/vald/internal/log"
	"github.com/vdaas/vald/internal/strings"
)

// Open opens the file with the given path, flag and permission.
// If the folder does not exists, create the folder.
// If the file does not exist, create the file.
func Open(path string, flg int, perm fs.FileMode) (file *os.File, err error) {
	if path == "" {
		return nil, errors.ErrPathNotSpecified
	}

	defer func() {
		if err != nil && file != nil {
			err = errors.Wrap(file.Close(), err.Error())
			file = nil
		}
	}()
	if ffi, err := os.Stat(path); err != nil {
		dir := filepath.Dir(path)
		fi, err := os.Stat(dir)
		if err != nil {
			err = MkdirAll(dir, perm)
			if err != nil {
				err = errors.ErrFailedToMkdir(err, dir, fi)
				log.Debug(err)
				return nil, err
			}
		}

		if flg&(os.O_CREATE|os.O_APPEND) > 0 {
			file, err = os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, perm)
			if err != nil {
				log.Debug(errors.ErrFailedToCreateFile(err, path, ffi))
			}
			if file != nil {
				err = file.Close()
				if err != nil {
					fi, ferr := file.Stat()
					if ferr == nil && fi != nil {
						err = errors.ErrFailedToCloseFile(err, path, fi)
					}
					log.Debug(err)
				}
			}
		}
	}

	file, err = os.OpenFile(path, flg, perm)
	if err != nil {
		err = errors.ErrFailedToOpenFile(err, path, flg, perm)
		log.Warn(err)
		return nil, err
	}

	return file, nil
}

func MoveDir(ctx context.Context, src, dst string) (err error) {
	return moveDir(ctx, src, dst, true)
}

func moveDir(ctx context.Context, src, dst string, rollback bool) (err error) {
	if len(src) == 0 || len(dst) == 0 || src == dst {
		return nil
	}
	exits, fi, err := exists(src)
	if !exits || !fi.IsDir() || err != nil {
		return errors.ErrDirectoryNotFound(err, src, fi)
	}

	err = os.Rename(src, dst)
	if err != nil {
		log.Debug(errors.ErrFailedToRenameDir(err, src, dst, nil, nil))
		var tmpPath string
		exits, fi, err := exists(dst)
		if exits && fi.IsDir() && err == nil {
			tmpPath = Join(filepath.Dir(dst), "tmp-"+strconv.FormatInt(fastime.UnixNanoNow(), 10))
			_ = os.RemoveAll(tmpPath)
			err = os.Rename(dst, tmpPath)
			defer os.RemoveAll(tmpPath)
			if err != nil {
				log.Debugf("err: %v\t now trying to move file with I/O copy and Remove old index", errors.ErrFailedToRenameDir(err, dst, tmpPath, fi, nil))
				err := CopyDir(ctx, dst, tmpPath)
				if err != nil {
					err = errors.ErrFailedToCopyDir(err, dst, tmpPath, nil, nil)
					log.Warn(err)
					return err
				}
				err = os.RemoveAll(dst)
				if err != nil && Exists(dst) {
					err = errors.ErrFailedToRemoveDir(err, dst, nil)
					if rollback {
						err = errors.Wrap(moveDir(ctx, tmpPath, dst, false), errors.Wrapf(err, "trying to recover temporary file %s to rollback previous operation", tmpPath).Error())
					}
					log.Warn(err)
					return err
				}
			}
			log.Debugf("directory %s successfully moved to tmp location %s", dst, tmpPath)
		}
		exits, fi, err = exists(src)
		if exits && fi.IsDir() && err == nil {
			err = os.Rename(src, dst)
			if err != nil {
				log.Debugf("err: %v\t now trying to move file with I/O copy and Remove old index", errors.ErrFailedToRenameDir(err, src, dst, fi, nil))
				err := CopyDir(ctx, src, dst)
				if err != nil {
					err = errors.ErrFailedToCopyDir(err, src, dst, fi, nil)
					if rollback {
						err = errors.Wrap(moveDir(ctx, tmpPath, dst, false), errors.Wrapf(err, "trying to recover temporary file %s to rollback previous operation", tmpPath).Error())
					}
					log.Warn(err)
					return err
				}
				err = os.RemoveAll(src)
				if err != nil && Exists(src) {
					err = errors.ErrFailedToRemoveDir(err, src, fi)
					log.Warn(err)
					return err
				}
			}
		}
	}
	log.Infof("directory %s successfully moved to destination directory %s", src, dst)
	return nil
}

func CopyDir(ctx context.Context, src, dst string) (err error) {
	if len(src) == 0 || len(dst) == 0 || src == dst {
		return nil
	}
	eg, _ := errgroup.New(ctx)
	err = filepath.WalkDir(src, func(childPath string, info fs.DirEntry, err error) error {
		if err != nil {
			fi, ierr := info.Info()
			if ierr != nil {
				err = errors.Wrap(err, ierr.Error())
			}
			err = errors.ErrFailedToWalkDir(err, src, childPath, nil, fi)
			log.Warn(err)
			return err
		}
		if src == childPath {
			return nil
		}
		dstPath := Join(dst, filepath.Base(childPath))
		if info.IsDir() {
			err = MkdirAll(dstPath, info.Type())
			if err != nil {
				log.Warn(errors.ErrFailedToMkdir(err, dstPath, nil))
			}
			return nil
		}
		eg.Go(func() (err error) {
			_, err = CopyFileWithPerm(ctx, childPath, dstPath, info.Type())
			return err
		})
		return nil
	})
	if err != nil {
		return errors.Wrap(eg.Wait(), err.Error())
	}
	return eg.Wait()
}

func CopyFile(ctx context.Context, src, dst string) (n int64, err error) {
	return CopyFileWithPerm(ctx, src, dst, fs.ModePerm)
}

func CopyFileWithPerm(ctx context.Context, src, dst string, perm fs.FileMode) (n int64, err error) {
	if len(src) == 0 || len(dst) == 0 || src == dst {
		return 0, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	defer func() {
		if err != nil {
			log.Warn(err)
		}
	}()

	exist, fi, err := exists(src)
	switch {
	case !exist || fi == nil || fi.Size() == 0:
		return 0, errors.Wrap(err, errors.ErrFileNotFound(src).Error())
	case err != nil && !os.IsExist(err):
		return 0, errors.Wrap(errors.ErrFileNotFound(src), err.Error())
	}
	flg := os.O_RDONLY | os.O_SYNC
	sf, err := Open(src, flg, perm)
	if err != nil || sf == nil {
		err = errors.ErrFailedToCopyDir(errors.ErrFailedToOpenFile(err, src, flg, perm), src, dst, fi, nil)
		return 0, err
	}
	defer func() {
		if sf != nil {
			derr := sf.Close()
			if derr != nil {
				err = errors.Wrap(err, errors.ErrFailedToCloseFile(derr, src, fi).Error())
			}
		}
	}()
	sr, err := io.NewReaderWithContext(ctx, sf)
	if err != nil {
		return 0, err
	}
	n, err = OverWriteFile(ctx, dst, sr, perm)
	if err != nil && !errors.Is(err, io.EOF) {
		err = errors.ErrFailedToCopyFile(err, src, dst, fi, nil)
		return 0, err
	}
	return n, nil
}

func WriteFile(ctx context.Context, target string, r io.Reader, perm fs.FileMode) (n int64, err error) {
	return writeFile(ctx, target, r, os.O_CREATE|os.O_WRONLY|os.O_SYNC, perm)
}

func OverWriteFile(ctx context.Context, target string, r io.Reader, perm fs.FileMode) (n int64, err error) {
	return writeFile(ctx, target, r, os.O_CREATE|os.O_TRUNC|os.O_WRONLY|os.O_SYNC, perm)
}

func AppendFile(ctx context.Context, target string, r io.Reader, perm fs.FileMode) (n int64, err error) {
	return writeFile(ctx, target, r, os.O_CREATE|os.O_APPEND|os.O_RDWR|os.O_SYNC, perm)
}

func writeFile(ctx context.Context, target string, r io.Reader, flg int, perm fs.FileMode) (n int64, err error) {
	if len(target) == 0 || r == nil {
		return 0, nil
	}

	exist, fi, err := exists(target)
	switch {
	case err == nil, exist, fi != nil && fi.Size() != 0:
		err = errors.ErrFileAlreadyExists(target)
	case err != nil && !os.IsNotExist(err):
		err = errors.Wrap(errors.ErrFileAlreadyExists(target), err.Error())
	}

	// open flag is not O_TRUNC or O_APPEND this function returns AlreadyExists error
	if err != nil && flg&(os.O_TRUNC|os.O_APPEND) <= 0 {
		return 0, err
	}

	f, err := Open(target, flg, fs.ModePerm)
	if err != nil || f == nil {
		err = errors.ErrFailedToOpenFile(err, target, flg, perm)
		return 0, err
	}
	defer func() {
		if f != nil {
			derr := f.Close()
			if derr != nil {
				err = errors.Wrap(err, errors.ErrFailedToCloseFile(derr, target, fi).Error())
			}
		}
	}()
	w, err := io.NewWriterWithContext(ctx, f)
	if err != nil {
		return 0, err
	}
	n, err = io.Copy(w, r)
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, err
	}
	err = f.Sync()
	if err != nil {
		return 0, err
	}
	return n, nil
}

// Exists returns file existence
func Exists(path string) (e bool) {
	e, _, _ = exists(path)
	return e
}

// ExistsWithDetail returns file existence
func ExistsWithDetail(path string) (e bool, fi fs.FileInfo, err error) {
	return exists(path)
}

// MkdirAll creates directory like mkdir -p
func MkdirAll(path string, perm fs.FileMode) (err error) {
	var (
		exist      bool
		fi         fs.FileInfo
		merr, rerr error
	)
	exist, fi, err = exists(path)
	if exist {
		if err == nil && fi != nil && fi.IsDir() {
			return nil
		}
		rerr = os.RemoveAll(path)
		if rerr != nil {
			err = errors.Wrap(err, rerr.Error())
		}
	}
	if err != nil {
		err = errors.ErrDirectoryNotFound(err, path, fi)
	}
	merr = os.MkdirAll(path, perm)
	if merr == nil {
		return nil
	}
	err = errors.Wrap(merr, err.Error())
	if err != nil {
		if os.IsPermission(err) {
			rerr = os.RemoveAll(path)
			if rerr != nil {
				err = errors.Wrap(err, errors.ErrFailedToRemoveDir(rerr, path, fi).Error())
			}
			merr = os.MkdirAll(path, fs.ModePerm)
			if merr != nil {
				err = errors.Wrap(err, errors.ErrFailedToMkdir(merr, path, fi).Error())
			}
		}
		log.Warn(err)
		return err
	}
	return nil
}

// MkdirTemp create temporary directory from given base path
// if base path is nil temporary directory will create from Go's standard library
func MkdirTemp(baseDir string) (path string, err error) {
	if len(baseDir) == 0 {
		baseDir = os.TempDir()
	}
	path = Join(baseDir, strconv.FormatInt(time.Now().UnixNano(), 10))
	err = MkdirAll(path, fs.ModePerm)
	if err != nil {
		err = errors.ErrFailedToMkTmpDir(err, path, nil)
		log.Debug(err)
		return "", err
	}
	return path, nil
}

// exists returns file existence with detailed information
func exists(path string) (exists bool, fi fs.FileInfo, err error) {
	fi, err = os.Stat(path)
	if err != nil {
		if os.IsExist(err) {
			return true, fi, nil
		}
		if os.IsNotExist(err) {
			return false, fi, err
		}
		return false, fi, err
	}
	return true, fi, nil
}

// ListInDir returns file list in directory
func ListInDir(path string) ([]string, error) {
	exists, fi, err := exists(path)
	if !exists {
		return nil, err
	}
	if fi.Mode().IsDir() && !strings.HasSuffix(path, string(os.PathSeparator)) {
		path += string(os.PathSeparator)
	}
	path = filepath.Dir(path)
	files, err := filepath.Glob(Join(path, "*"))
	if err != nil {
		return nil, err
	}
	return files, nil
}

func Join(paths ...string) (path string) {
	if paths == nil {
		return ""
	}
	if len(paths) > 1 {
		path = join(paths...)
	} else {
		path = paths[0]
	}
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}

	root, err := os.Getwd()
	if err != nil {
		err = errors.ErrFailedToGetAbsPath(err, path)
		log.Warn(err)
		return filepath.Clean(path)
	}
	return filepath.Clean(join(root, path))
}

func join(paths ...string) (path string) {
	for i, path := range paths {
		if path != "" {
			return strings.Join(paths[i:], string(os.PathSeparator))
		}
	}
	return ""
}
