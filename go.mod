module github.com/vdaas/vald

go 1.16

replace (
	cloud.google.com/go => cloud.google.com/go v0.89.1-0.20210802202428-d571c6f4337e
	cloud.google.com/go/storage => cloud.google.com/go/storage v1.16.1-0.20210802202428-d571c6f4337e
	github.com/Azure/go-autorest => github.com/Azure/go-autorest v14.2.1-0.20210701162346-76c7860e9b60+incompatible
	github.com/aws/aws-sdk-go => github.com/aws/aws-sdk-go v1.40.13
	github.com/boltdb/bolt => github.com/boltdb/bolt v1.3.1
	github.com/chzyer/logex => github.com/chzyer/logex v1.1.11-0.20170329064859-445be9e134b2
	github.com/coreos/etcd => go.etcd.io/etcd v3.3.25+incompatible
	github.com/docker/docker => github.com/moby/moby v20.10.7+incompatible
	github.com/envoyproxy/protoc-gen-validate => github.com/envoyproxy/protoc-gen-validate v0.6.1
	github.com/go-sql-driver/mysql => github.com/go-sql-driver/mysql v1.6.0
	github.com/gocql/gocql => github.com/gocql/gocql v0.0.0-20210707082121-9a3953d1826d
	github.com/gogo/googleapis => github.com/gogo/googleapis v1.4.1
	github.com/gogo/protobuf => github.com/gogo/protobuf v1.3.2
	github.com/google/go-cmp => github.com/google/go-cmp v0.5.6
	github.com/google/pprof => github.com/google/pprof v0.0.0-20210726183535-c50bf4fe5303
	github.com/googleapis/gnostic => github.com/googleapis/gnostic v0.5.5
	github.com/gophercloud/gophercloud => github.com/gophercloud/gophercloud v0.19.0
	github.com/gorilla/websocket => github.com/gorilla/websocket v1.4.2
	github.com/hailocab/go-hostpool => github.com/kpango/go-hostpool v0.0.0-20210303030322-aab80263dcd0
	github.com/klauspost/compress => github.com/klauspost/compress v1.13.3-0.20210802161833-6c2ac28c4d87
	github.com/kpango/glg => github.com/kpango/glg v1.6.4
	github.com/tensorflow/tensorflow => github.com/tensorflow/tensorflow v2.1.2+incompatible
	github.com/zeebo/xxh3 => github.com/zeebo/xxh3 v0.10.0
	go.opentelemetry.io/otel => go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/internal/metric => go.opentelemetry.io/otel/internal/metric v0.22.0
	go.opentelemetry.io/otel/metric => go.opentelemetry.io/otel/metric v0.22.0
	go.opentelemetry.io/otel/sdk => go.opentelemetry.io/otel/sdk v0.20.0
	go.opentelemetry.io/otel/trace => go.opentelemetry.io/otel/trace v0.20.0
	go.opentelemetry.io/proto/otlp => go.opentelemetry.io/proto/otlp v0.9.0
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20210711020723-a769d52b0f97
	golang.org/x/exp => golang.org/x/exp v0.0.0-20210729172720-737cce5152fc
	golang.org/x/image => golang.org/x/image v0.0.0-20210628002857-a66eb6448b8d
	golang.org/x/lint => golang.org/x/lint v0.0.0-20210508222113-6edffad5e616
	golang.org/x/mobile => golang.org/x/mobile v0.0.0-20210716004757-34ab1303b554
	golang.org/x/mod => golang.org/x/mod v0.4.2
	golang.org/x/net => golang.org/x/net v0.0.0-20210726213435-c6fcb2dbf985
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20210628180205-a41e5a781914
	golang.org/x/sync => golang.org/x/sync v0.0.0-20210220032951-036812b2e83c
	golang.org/x/sys => golang.org/x/sys v0.0.0-20210630005230-0f9fa26af87c
	golang.org/x/term => golang.org/x/term v0.0.0-20210615171337-6886f2dfbf5b
	golang.org/x/text => golang.org/x/text v0.3.6
	golang.org/x/time => golang.org/x/time v0.0.0-20210723032227-1f47c861a9ac
	golang.org/x/tools => golang.org/x/tools v0.1.5
	golang.org/x/xerrors => golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1
	google.golang.org/api => google.golang.org/api v0.52.0
	google.golang.org/appengine => google.golang.org/appengine v1.6.7
	google.golang.org/grpc => google.golang.org/grpc v1.39.0
	google.golang.org/grpc/cmd/protoc-gen-go-grpc => google.golang.org/grpc/cmd/protoc-gen-go-grpc v1.1.0
	google.golang.org/protobuf => google.golang.org/protobuf v1.27.1
	honnef.co/go/tools => honnef.co/go/tools v0.2.0
	k8s.io/api => k8s.io/api v0.21.3
	k8s.io/apiextensions-apiserver => k8s.io/apiextensions-apiserver v0.21.3
	k8s.io/apimachinery => k8s.io/apimachinery v0.21.3
	k8s.io/cli-runtime => k8s.io/cli-runtime v0.21.3
	k8s.io/client-go => k8s.io/client-go v0.21.3
	k8s.io/metrics => k8s.io/metrics v0.21.3
	sigs.k8s.io/controller-runtime => sigs.k8s.io/controller-runtime v0.9.5
)

require (
	cloud.google.com/go v0.88.0
	cloud.google.com/go/storage v1.15.0
	code.cloudfoundry.org/bytefmt v0.0.0-20210608160410-67692ebc98de
	contrib.go.opencensus.io/exporter/jaeger v0.2.1
	contrib.go.opencensus.io/exporter/prometheus v0.3.0
	contrib.go.opencensus.io/exporter/stackdriver v0.13.8
	github.com/aws/aws-sdk-go v1.38.35
	github.com/aws/aws-sdk-go-v2 v1.7.1
	github.com/aws/aws-sdk-go-v2/config v1.5.0
	github.com/aws/aws-sdk-go-v2/feature/s3/manager v1.3.2
	github.com/aws/aws-sdk-go-v2/service/s3 v1.11.1
	github.com/aws/smithy-go v1.6.0
	github.com/cespare/xxhash/v2 v2.1.1
	github.com/envoyproxy/protoc-gen-validate v0.1.0
	github.com/fsnotify/fsnotify v1.4.9
	github.com/go-redis/redis/v8 v8.11.1
	github.com/go-sql-driver/mysql v1.6.0
	github.com/go-toolsmith/strparse v1.0.0 // indirect
	github.com/gocql/gocql v0.0.0-20200131111108-92af2e088537
	github.com/gocraft/dbr/v2 v2.7.2
	github.com/gogo/googleapis v0.0.0-20180223154316-0cd9801be74a
	github.com/gogo/protobuf v1.3.2
	github.com/gogo/status v1.1.0
	github.com/golang/protobuf v1.5.2
	github.com/google/go-cmp v0.5.6
	github.com/google/gofuzz v1.2.0
	github.com/gorilla/mux v1.8.0
	github.com/hashicorp/go-version v1.3.0
	github.com/json-iterator/go v1.1.11
	github.com/klauspost/compress v1.12.2
	github.com/kpango/fastime v1.0.17
	github.com/kpango/fuid v0.0.0-20210407064122-2990e29e1ea5
	github.com/kpango/gache v1.2.6
	github.com/kpango/glg v1.6.2
	github.com/lucasb-eyer/go-colorful v1.2.0
	github.com/pierrec/lz4/v3 v3.3.2
	github.com/quasilyte/go-ruleguard v0.3.7
	github.com/quasilyte/go-ruleguard/dsl v0.3.6
	github.com/scylladb/gocqlx v1.5.0
	github.com/tensorflow/tensorflow v0.0.0-00010101000000-000000000000
	github.com/yahoojapan/gongt v0.0.0-20190517050727-966dcc7aa5e8
	github.com/yahoojapan/ngtd v0.0.0-20200424071638-9872bbae3700
	go.opencensus.io v0.23.0
	go.uber.org/automaxprocs v1.4.0
	go.uber.org/goleak v1.1.10
	go.uber.org/zap v1.18.1
	gocloud.dev v0.23.0
	golang.org/x/net v0.0.0-20210614182718-04defd469f4e
	golang.org/x/oauth2 v0.0.0-20210628180205-a41e5a781914
	golang.org/x/sync v0.0.0-20210220032951-036812b2e83c
	golang.org/x/sys v0.0.0-20210630005230-0f9fa26af87c
	golang.org/x/tools v0.1.5
	gonum.org/v1/hdf5 v0.0.0-20210714002203-8c5d23bc6946
	gonum.org/v1/plot v0.9.0
	google.golang.org/api v0.51.0
	google.golang.org/genproto v0.0.0-20210729151513-df9385d47c1b
	google.golang.org/grpc v1.39.0
	gopkg.in/yaml.v2 v2.4.0
	inet.af/netaddr v0.0.0-20210729200904-31d5ee66059c
	k8s.io/api v0.21.3
	k8s.io/apimachinery v0.21.3
	k8s.io/cli-runtime v0.0.0-00010101000000-000000000000
	k8s.io/client-go v0.21.3
	k8s.io/metrics v0.0.0-00010101000000-000000000000
	sigs.k8s.io/controller-runtime v0.0.0-00010101000000-000000000000
)
