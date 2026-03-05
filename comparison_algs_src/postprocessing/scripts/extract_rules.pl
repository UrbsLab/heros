#!/usr/bin/perl -w

my $trobat=0;

while(<STDIN>) {
	chomp;
	if(/^Phenotype/) {
		$trobat=1;
		next;
	}
	if($trobat) {
		if(/^[0-9]/) {
			s/[0-9]+://g;
			print "$_\n";
		} else {
			$trobat=0;
		}
	}
}

