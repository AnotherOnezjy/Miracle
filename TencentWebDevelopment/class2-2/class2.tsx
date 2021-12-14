function average(list: number[]) {
	let res = 0;
	for (let i = 0; i < list.length; i++) res += list[i];
	res /= list.length;
	return res;
}

console.log(average([87, 88, 86, 82, 85, 89, 87, 87, 90, 90, 88, 87]));
